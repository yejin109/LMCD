import os
import datetime
import wandb
import yaml
import torch

credential = yaml.load(open('./credential.yml'), Loader=yaml.FullLoader)
wandb.login(key=credential['wand']['key'])

os.environ['WANDB_ENTITY'] = 'yejin109/lmcd'
os.environ['WANDB_WATCH'] = 'all'
os.environ['WANDB_PROJECT'] = 'SRS'

os.environ['ITERATION_STEP'] = str(0)
os.environ['EXP_NAME'] = '-'.join(
    ['lmcd', os.environ['WANDB_PROJECT'], str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))])

os.environ['LOG_DIR'] = f'./logs/{os.environ["EXP_NAME"]}'
os.mkdir(os.environ['LOG_DIR'])
os.mkdir(os.path.join(os.environ['LOG_DIR'], 'batch'))
os.environ['VERBOSE'] = "0"
os.environ['DEVICE'] = 'cuda:0'

torch.backends.cudnn.benchmark = True
