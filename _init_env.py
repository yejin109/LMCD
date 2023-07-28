import os
import datetime
import wandb
import yaml
import torch

credential = yaml.load(open('./credential.yml'), Loader=yaml.FullLoader)
wandb.login(key=credential['wandb']['key'])

os.environ['WANDB_ENTITY'] = 'yejin109/lmcd'
os.environ['WANDB_WATCH'] = 'all'

os.environ['VERBOSE'] = "0"
os.environ['DEVICE'] = 'cuda:0'

torch.backends.cudnn.benchmark = True
