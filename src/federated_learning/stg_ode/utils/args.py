import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--num_users', type=int, default=100) #16000 -- 10100 3000
parser.add_argument('--frac', type=float, default=0.01, help="the fraction of clients: C")

parser.add_argument('--remote', action='store_true', help='the code run on a server')
parser.add_argument('--num-gpu', type=int, default=0, help='the number of the gpu to use')
parser.add_argument('--epochs', type=int, default=3, help='train epochs')
parser.add_argument('--batch', type=int, default=1, help='batch size')

parser.add_argument('--torch-seed', type=int, default=0, help='torch_seed')
parser.add_argument('--model', type=str, default='stgode', help='model name')

parser.add_argument('--filename', type=str, default='pems04')
parser.add_argument('--train_ratio', type=float, default=0.6, help='the ratio of training dataset')
parser.add_argument('--valid_ratio', type=float, default=0.2, help='the ratio of validating dataset')
parser.add_argument('--his_length', type=int, default=12, help='the length of history time series of input')
parser.add_argument('--pred_length', type=int, default=12, help='the length of target time series for prediction')

parser.add_argument('--dataset', type=str, default='pems04', help="name of dataset")
parser.add_argument('--sigma1', type=float, default=0.1, help='sigma for the semantic matrix')
parser.add_argument('--sigma2', type=float, default=10, help='sigma for the spatial matrix')
parser.add_argument('--thres1', type=float, default=0.6, help='the threshold for the semantic matrix')
parser.add_argument('--thres2', type=float, default=0.5, help='the threshold for the spatial matrix')
parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')

parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
parser.add_argument('--iid',default='iid', action='store_true', help='whether i.i.d or not')
parser.add_argument('--verbose', action='store_true', help='verbose print')  ##详细打印
parser.add_argument('--gpu', type=int, default=-1, help="GPU ID, -1 for CPU")
parser.add_argument('--log', action='store_true', help='if write log to files')
args = parser.parse_args()
