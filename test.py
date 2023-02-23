
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import nn
import torch
from tqdm import tqdm
from eval import masked_mae_np, masked_mape_np, masked_rmse_np

# loss_fn=nn.SmoothL1Loss()

@torch.no_grad()
def test_net(loader, model, std, mean, device, args):
    batch_rmse_loss = 0  
    batch_mae_loss = 0
    batch_mape_loss = 0
    data_loader = DataLoader(loader, batch_size=args.batch)
    for idx, (inputs, targets) in enumerate(tqdm(data_loader)):
        model.eval()
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

    return batch_rmse_loss / (idx + 1), batch_mae_loss / (idx + 1), batch_mape_loss / (idx + 1)

