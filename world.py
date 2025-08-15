import torch
import argparse
import yaml

def load_arguments_from_yaml(filename):
    with open(filename, 'r') as file:
        config = yaml.safe_load(file)
    return config

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--core', type=int, default=0)
parser.add_argument('--dataset', type=str, default='ciao', help='available datasets: [ciao]')
parser.add_argument('--epoches', type=int, default=1000, help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--test_batch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--ui_n_layers', type=int, default=2, help='number of layers for UI encoder')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--ui_dropout', type=bool, default=False)
parser.add_argument('--keep_prob_ui', type=float, default=0.9, help='keep probability')

parser.add_argument('--schedule_type', type=str, default='', help='available schedule_type: [linear]')
parser.add_argument('--sigma_min', type=float, default=0.001)
parser.add_argument('--sigma_max', type=float, default=1.0)
parser.add_argument('--num_step', type=int, default=30, help='discrete diffusion step')
parser.add_argument('--item_freq_lambda', type=float, default=0.5)
parser.add_argument('--gaussian_scale', type=float, default=6.0)
parser.add_argument('--consistency_mode', type=str, default='freq_inference', help='available: [freq_inference, euler_solver, p_reverse]')
parser.add_argument('--consistency_scale', type=int, default=10, help='num_step/consistency_scale')
parser.add_argument('--consistency_loss_lambda', type=float, default=0.01, help='consistency_loss_lambda')
parser.add_argument('--beta_consistency', type=float, default=0.95, help='beta_consistency')

parser.add_argument('--consistency_loss_type', type=str, default=10, help='l2')
parser.add_argument('--dpo_loss_type', type=str, help='l2')
parser.add_argument('--dpo_mode', type=str, help='available: [neg, neg_sample]')
parser.add_argument('--dpo_gamma', type=float)
parser.add_argument('--dpo_lamda', type=float)

parser.add_argument('--aggregation_type', type=str, default='sum', help='available: [sum, mean]')
parser.add_argument('--timestep', type=str, default='layerwise', help='available: [layerwise, token, none]')
parser.add_argument('--sampling_scale', type=int, default=0)
parser.add_argument('--sampling_type', type=str, default='first', help='available: [first, agg]')
parser.add_argument('--sampling_num_step', type=int, default=50)

parser.add_argument('--ema_rate', type=float, default=10, help='0.999')
parser.add_argument('--tau', type=float, default=0.4, help='0.4')
parser.add_argument('--ssl_loss_lambda', type=float, default=1.0, help='ssl_loss_lambda')

args = parser.parse_args()

device = (
        torch.device("cuda:"+str(args.core))
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

config = load_arguments_from_yaml(f'configures/{args.dataset}.yaml')
for arg, value in config.items():
    setattr(args, arg, value)
    
seed = args.seed
dataset = args.dataset
diffusion_type = "absorb"
topks = eval(args.topks)
pretrain_epoch = 100  # 70

config_denoiser = {}
config_denoiser['nhead'] = 4
config_denoiser['dim_feedforward'] = 2048
config_denoiser['num_layers'] = 5

config_ssl = {}
config_ssl['tau'] = args.tau

config_sampling = {}
config_sampling['sampling_from_noise'] = True
config_sampling['sampling_scale'] = args.sampling_scale
config_sampling['sampling_num_step'] = args.sampling_num_step
config_sampling['sampling_type'] = args.sampling_type
# 'first', 'agg'



# config_dpo = {}
# config_dpo['dpo_mode'] = args.dpo_mode
# config_dpo['dpo_gamma'] = args.dpo_gamma
# config_dpo['dpo_lamda'] = args.dpo_lamda