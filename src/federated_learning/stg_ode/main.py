import matplotlib.pyplot as plt
import copy
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import numpy as np
import torch


from utils.sampling import divide_iid,divide_noniid
from models.Update import LocalUpdate
from models.Fed import FedAvg
from models.model import ODEGCN

from utils.args import args
from utils.utils import generate_dataset, read_data, get_normalized_adj
from test import test_net

from loguru import logger



if __name__ == '__main__':

    # Setting the device type
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    if args.log:
        logger.add('log_{time}.log')
    options = vars(args)
    if args.log:
        logger.info(options)
    else:
        print(options)

    

#----------------------------------- Load dataset and split users --------------------------------------#

    # The Size of t Over all training dataset has the follows shapes
    # PEMS04 == shape: (16992, 307, 3)    feature: flow,occupy,speed
    # PEMSD7M == shape: (12672, 228, 1)
    # PEMSD7L == shape: (12672, 1026, 1)
    
    # The train dataset has a shape of (10172, 307, 3)
    # The valid dataset has a shape of (3375, 307, 3)
    # The test dataset has a shape of (3376, 307, 3)
    data, mean, std, dtw_matrix, sp_matrix = read_data(args)
    train_dataset, valid_dataset, test_dataset = generate_dataset(data, args)
   
  
    # Normalize Adjacency matrix
    A_sp_wave = get_normalized_adj(sp_matrix).to(args.device)
    A_se_wave = get_normalized_adj(dtw_matrix).to(args.device)

    # Sample users generation by independent and identically distributed devices (I.I.D) and non-I.I.D 
    if args.iid:
        dict_users = divide_iid(train_dataset, args.num_users)
     
    else:
        dict_users = divide_noniid(train_dataset, args.num_users)
        
        
#----------------------------------- Building Model --------------------------------------#

    # Create a model Architecuter for Spatial-Temporal Graph ODE Networks
    net_glob = ODEGCN(num_nodes=data.shape[1], 
                      num_features=data.shape[2], 
                      num_timesteps_input=args.his_length, 
                      num_timesteps_output=args.pred_length, 
                      A_sp_hat=A_sp_wave, 
                      A_se_hat=A_se_wave)
    
    # Set the Hyper-parameters, define the optimizer and scheduler for the Global Model
    lr = args.lr
    optimizer = torch.optim.AdamW(net_glob.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    # Set the model to train and send it to device.
    net_glob = net_glob.to(args.device)
    net_glob.train()
    

    # copy weights
    w_glob = net_glob.state_dict() #Get network parameters
 
    # training 
    loss_train = []
    acc_train = []

    for iter in range(args.epochs):
        
        w_locals, loss_locals = [], []
        m = max(int(args.frac * args.num_users), 1)                                              # Sets the max limit for the idx_user
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)                   # Randomly choices value
        
        # print(len(idxs_users))
        # print(idxs_users)
        
        for idx in idxs_users:
            print('Training...')
            local = LocalUpdate(args=args, dataset=train_dataset, idxs=dict_users[idx])
            
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))

        scheduler.step()         # Decays the learning rate of each parameter group by gamma=0.5 every step_size epochs.

        # update global weights
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)
        acc_train.append(1-loss_avg)



#----------------------------------- Testing the Model  --------------------------------------#


    test_rmse, test_mae, test_mape = test_net(test_dataset, net_glob, std, mean, args.device, args)
    print(f'##on test data## rmse loss: {test_rmse}, mae loss: {test_mae}, mape loss: {test_mape}')
    
    
#----------------------------------- Plot the Loss Graph --------------------------------------#



    # plt.figure()
    # plt.plot(range(len(loss_train)), loss_train)
    # plt.ylabel('train_loss')
    # plt.savefig('./save/fed_loss{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # plt.figure()
    # plt.plot(range(len(loss_train)), acc_train)
    # plt.ylabel('train_accuracy')
    # plt.savefig('./save/fed_accuracy{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))
    # plt.show()        
    

