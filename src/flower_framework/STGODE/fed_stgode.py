from collections import OrderedDict
from typing import List, Tuple

from flwr.common import Metrics

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR

import pickle
import numpy as np
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
import torch

from model import ODEGCN
from utils import MyDataset, read_data, get_normalized_adj
from eval import masked_mae_np, masked_mape_np, masked_rmse_np

from torch.utils.data import DataLoader, random_split

import ray
import flwr as fl
if torch.cuda. is_available():
    print("Using Cuda!")
    DEVICE = torch.device("cuda")
else:
    print("Using CPU")
    DEVICE = torch.device("cpu")



class Args:
    def __init__(self):
        self.remote = False
        self.num_gpu = 0
        self.epochs = 7
        self.batch_size = 16
        self.batch = 16
        self.frac = 0.1
        self.num_users = 100
        self.filename = 'pems08'
        self.train_ratio = 0.6
        self.valid_ratio = 0.2
        self.his_length = 12
        self.pred_length = 12
        self.sigma1 = 0.1
        self.sigma2 = 10
        self.thres1 = 0.6
        self.thres2 = 0.5
        self.lr = 2e-3
        self.log = False

args = Args()



def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    #print('works till here', len(state_dict))
    net.load_state_dict(state_dict, strict=True)



def train(loader, model, optimizer, criterion, epochs, scheduler, std, mean,device):
    print('training for epochs:', epochs)
    epoch_loss = []
    model.train()
    for epoch in range(1, epochs+1):
        print('epoch#:', epoch)
        batch_loss = 0
        for idx, (inputs, targets) in enumerate(loader):
            
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            batch_loss += loss.detach().cpu().item() 
        epoch_loss.append(batch_loss)
        loss = batch_loss
       
        train_rmse, train_mae, train_mape = eval(loader, model, std, mean, device)

        if args.log:
            logger.info(f'\n##on train data## loss: {loss}, \n' + 
                        f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n')
        else:
            print(f'\n##on train data## loss: {loss}, \n' + 
                f'##on train data## rmse loss: {train_rmse}, mae loss: {train_mae}, mape loss: {train_mape}\n')
        
    scheduler.step()

    return sum(epoch_loss)/epochs


@torch.no_grad()
def eval(loader, model, std, mean, device):
    batch_rmse_loss = 0  
    batch_mae_loss = 0
    batch_mape_loss = 0
    print('evaluating')
    model.eval()
    for idx, (inputs, targets) in enumerate(loader):

        inputs = inputs.to(device)
        targets = targets.to(device)
        output = model(inputs)
        
        out_unnorm = output.detach().cpu().numpy()*std + mean
        target_unnorm = targets.detach().cpu().numpy()*std + mean

        mae_loss = masked_mae_np(target_unnorm, out_unnorm, 0)
        rmse_loss = masked_rmse_np(target_unnorm, out_unnorm, 0)
        mape_loss = masked_mape_np(target_unnorm, out_unnorm, 0)
        batch_rmse_loss += rmse_loss
        batch_mae_loss += mae_loss
        batch_mape_loss += mape_loss
    print("eval rmse loss: ", batch_rmse_loss)

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1)



def generate_dataset(data, args):
    """
    Args:
        data: input dataset, shape like T * N
        batch_size: int 
        train_ratio: float, the ratio of the dataset for training
        his_length: the input length of time series for prediction
        pred_length: the target length of time series of prediction

    Returns:
        train_dataloader: torch tensor, shape like batch * N * his_length * features
        test_dataloader: torch tensor, shape like batch * N * pred_length * features
    """
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    valid_ratio = args.valid_ratio
    his_length = args.his_length
    pred_length = args.pred_length
    train_dataset = MyDataset(data, 0, data.shape[0] * train_ratio, his_length, pred_length)

    valid_dataset = MyDataset(data, data.shape[0]*train_ratio, data.shape[0]*(train_ratio+valid_ratio), his_length, pred_length)

    test_dataset = MyDataset(data, data.shape[0]*(train_ratio+valid_ratio), data.shape[0], his_length, pred_length)

    return train_dataset, valid_dataset, test_dataset



class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, optimiser, schedular, learning_rate, epochs, loss_function, mean, std ):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.optimiser = torch.optim.AdamW(net.parameters(), lr=self.learning_rate)
        self.scheduler = StepLR(self.optimiser, step_size=50, gamma=0.5)
        self.mean = mean
        self.std = std

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        loss = train(self.trainloader, self.net, self.optimiser ,  self.loss_function, self.epochs , self.scheduler, self.std, self.mean, DEVICE)
        return get_parameters(self.net), len(self.trainloader)*self.trainloader.batch_size, {"train_loss":loss}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        valid_rmse, valid_mae, valid_mape = eval(self.valloader, self.net, self.std, self.mean, DEVICE)
        return valid_rmse, len(self.valloader)*self.valloader.batch_size , {'valid_mape':valid_mape, 'valid_mae':valid_mae, 'valid_rmse':valid_rmse}
    


def fit_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    loss = [num_examples * m["train_loss"] for num_examples, m in metrics]
    loss_client = [m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {'client_metrics': {"loss":loss_client}, 'examples':examples,'average_metrics':{'average loss': sum(loss)/sum(examples)}}
    

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used

    valid_rmse = [num_examples * m["valid_rmse"] for num_examples, m in metrics]
    valid_mae = [num_examples * m["valid_mae"] for num_examples, m in metrics]
    valid_mape = [num_examples * m["valid_mape"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    valid_rmse_client = [m["valid_rmse"] for num_examples, m in metrics]
    valid_mae_client = [m["valid_mae"] for num_examples, m in metrics]
    valid_mape_client = [m["valid_mape"] for num_examples, m in metrics]


    # Aggregate and return custom metric (weighted average)
    return {'average_metrics':{'valid_mape':sum(valid_mape)/sum(examples), 'valid_mae':sum(valid_mae)/sum(examples), 'valid_rmse':sum(valid_rmse)/sum(examples)},
            'client_metrics': {'valid_mape':valid_mape_client, 'valid_mae':valid_mae_client, 'valid_rmse':valid_rmse_client, 'examples': examples}}

def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    epochs = 2
    # Load model
    learning_rate = lr
    net = ODEGCN(num_nodes=data.shape[1], 
            num_features=data.shape[2], 
            num_timesteps_input=args.his_length, 
            num_timesteps_output=args.pred_length, 
            A_sp_hat=A_sp_wave, 
            A_se_hat=A_se_wave).to(DEVICE)
    #net.std = std
    #net.mean = mean

    w_glob = net.state_dict()
    
    net.load_state_dict(w_glob, strict=True)
    net(torch.rand(args.batch_size, feature_size, 12, 3).to(DEVICE))
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(cid = cid, net=net, trainloader=trainloader, valloader=valloader, optimiser=None , schedular=None,learning_rate=learning_rate, 
                        epochs=epochs , loss_function= criterion , mean=mean, std=std)


#params = get_parameters(ShallowRegressionLSTM(input_size=train_x.shape[1], hidden_units=num_hidden_units))

# Create FedAvg strategy

if args.log:
    logger.add('log_{time}.log')
options = vars(args)
if args.log:
    logger.info(options)
else:
    print(options)


epoch = args.epochs
NUM_CLIENTS = 5

data, mean, std, dtw_matrix, sp_matrix = read_data(args)
train_loader, valid_loader, test_loader = generate_dataset(data, args)
A_sp_wave = get_normalized_adj(sp_matrix).to(DEVICE)
A_se_wave = get_normalized_adj(dtw_matrix).to(DEVICE)

batch_size = args.batch_size
train_loader = Subset(train_loader, np.arange(1000))
valid_loader = Subset(valid_loader, np.arange(200))
feature_size = train_loader.dataset.data.shape[1]
inds = np.array_split(np.random.randint(len(train_loader), size=len(train_loader)), NUM_CLIENTS)
trainloaders = []
valloaders = []
for idx in inds:
    trainloaders.append(DataLoader(Subset(train_loader, idx), batch_size=batch_size, shuffle=True))


inds = np.array_split(np.random.randint(len(valid_loader), size=len(valid_loader)), NUM_CLIENTS)
for idx in inds:
    valloaders.append(DataLoader(Subset(valid_loader, idx), batch_size=batch_size, shuffle=True))

print("total train loaders:", len(trainloaders))
print("total  val loaders:", len(valloaders))
print("len of single train loader:", len(trainloaders[0]))
print("len of single val loader:", len(valloaders[0]))
print('feature size', feature_size)



lr = args.lr
#optimizer = torch.optim.AdamW(net.parameters(), lr=lr)
criterion = nn.SmoothL1Loss()

#best_valid_rmse = 1000 
#scheduler = StepLR(optimizer, step_size=50, gamma=0.5)



strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=5,  # Never sample less than 10 clients for training
    min_evaluate_clients=5,  # Never sample less than 5 clients for evaluation
    min_available_clients=5,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn =  weighted_average,
    fit_metrics_aggregation_fn = fit_average
)

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1}

# Start simulation
flower_history = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
    client_resources=client_resources,
)

with open('flower_history.pkl', 'wb') as f:
    pickle.dump(flower_history, f)