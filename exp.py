import argparse
from utils import seed_everything, get_device
from trainer import Trainer

def run(args_dict):
    seed_everything(args_dict['seed'])
    device = get_device(args_dict['device'])

    trainer = Trainer(args_dict, args_dict['seed'], device)

    trainer.fit(
        args_dict['epoch'],
        args_dict['optimizer'],
        args_dict['lr'],
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='hit', help='Name of Model (for embedding)')

    parser.add_argument('--go', action='store_false')
    parser.add_argument('--hpo', action='store_false')
    parser.add_argument('--do', action='store_false')

    parser.add_argument('--y_dim', type=int, default=3)

    parser.add_argument('--depth_g', type=int, default=2)
    parser.add_argument('--depth_d', type=int, default=2)

    parser.add_argument('--dim_in', type=int, default=32, help='Input dimension')
    parser.add_argument('--dim_hidden', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--dim_out', type=int, default=32, help='Output dimension')
    
    parser.add_argument('--n_heads', type=int, default=1, help='Number of hops')

    parser.add_argument('--n_hop', type=int, default=3, help='Number of hops')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout')

    parser.add_argument('--depth', type=int, default=2, help='Depth (layers) of backbone model')
    
    parser.add_argument('--epoch', type=int, default=50, help='Training Epochs')
    parser.add_argument('--batch_size', type=int, default=5000, help='Batch size')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer name')
    parser.add_argument('--lr', type=float, default= 1e-2, help='Learning rate')
    # hit : 1e-2   
    parser.add_argument('--w_decay', type=float, default=None, help='l2 reg, weight decay')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--device', type=int, default=0, choices=[-1, 0, 1], help='device information (-1:cpu, 0~1: cuda num')

    args = parser.parse_args()

    args_dict = vars(args)

    run(args_dict)
