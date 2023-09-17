import datetime
import os
import argparse

import torch.nn as nn
import torch
from data import get_dataset, get_data, CustomMLMCollator
from _utils import CustomWandbCallback, AscMaskCallBack, AdaMaskCallBack
import numpy as np
from transformers import AutoModelForMaskedLM, TrainingArguments, AutoTokenizer, AutoConfig
from sklearn.metrics import accuracy_score
from huggingface import CustomTrainer
from accelerate import init_empty_weights


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default="prajjwal1/bert-tiny")
parser.add_argument('--ckpt', default=None)

parser.add_argument('--data', default='bookcorpus', required=False, help='default bookcorpus, wikipedia')
parser.add_argument('--use_partial_data', default=True, required=False)
parser.add_argument('--partial_data_size', default=4, type=int, required=False)

parser.add_argument('--chunk_size', default=64, type=int)

# train
parser.add_argument('--lr', default=2e-5)
parser.add_argument('--wd', default=1e-2)
parser.add_argument('--epochs', default=1)
parser.add_argument('--b_train', default=128, type=int)
parser.add_argument('--max_steps', type=int, default=20000)

parser.add_argument('--p', default=.20, type=float)
parser.add_argument('--ada_mask', default=True, required=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--entropy', default=False, required=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--mrd', default=False, required=False, action=argparse.BooleanOptionalAction)

# Validation
parser.add_argument('--b_eval', default=256, type=int)
parser.add_argument('--shard_eval', default=300, type=int)

# Test
parser.add_argument('--test', default=False, required=False, action=argparse.BooleanOptionalAction)

# Log
parser.add_argument('--logging_steps', type=int, default=100)
parser.add_argument('--save_steps', type=int, default=2000)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CACHE_DIR'] = 'C:/Users/ay011/.cache/huggingface/datasets'
    os.environ['WANDB_PROJECT'] = args.data + ' - v6'
    os.environ['EXP_NAME'] = '-'.join(
        ['lmcd', os.environ['WANDB_PROJECT'], str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))])

    os.environ['LOG_DIR'] = f'./logs/{os.environ["EXP_NAME"]}'
    os.mkdir(os.environ['LOG_DIR'])
    os.mkdir(os.path.join(os.environ['LOG_DIR'], 'batch'))

    np.random.seed(args.seed)

    _model_path = args.ckpt if args.ckpt is not None else args.model_type
    _model_kwargs = {
        '_fast_init': False
    }
    # V6 : Randomly initialize model
    config = AutoConfig.from_pretrained(_model_path)
    model = AutoModelForMaskedLM.from_config(config)
    train_args = {'learning_rate': args.lr, 'num_train_epochs': args.epochs, 'weight_decay': args.wd,
                  'per_device_train_batch_size': args.b_train, 'per_device_eval_batch_size': args.b_eval,
                  'do_eval': True, 'do_train': True,
                  # 'max_steps': args.max_steps,
                  'logging_steps': args.logging_steps, 'save_steps': args.save_steps}

    if args.data_type == 'synthetic':
        data = get_data((args.v, args.dist, args.n, args.D, args.b, args.seed))
        dataset = get_dataset(args.data_type, data.get_data())
        dataset.set_format(type='torch', columns=['input_ids'])
        dataset = dataset.train_test_split(args.test_ratio)
        tokenizer = None
    elif args.data_type == 'huggingface':
        dataset = get_dataset(args.data_type, args.data)
        if args.use_partial_data:
            dataset['train'] = dataset['train'].shard(args.partial_data_size, index=0)
        if 'unsupervised' in dataset.keys():
            del dataset['unsupervised']
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        tokenized_datasets = dataset.map(tokenize_function,
                                         batched=True, remove_columns=list(dataset['train'].features.keys()),
                                         fn_kwargs={'_tokenizer': tokenizer})
        chunk_size = args.chunk_size
        lm_datasets = tokenized_datasets.map(group_texts,
                                             batched=True,
                                             fn_kwargs={'_chunk_size': chunk_size})
        lm_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        dataset = lm_datasets
        tokenized_datasets = None

        if isinstance(dataset, dict) and ('test' not in dataset.keys()):
            dataset = dataset['train'].train_test_split(0.01)

        mlm_collator = CustomMLMCollator(tokenizer, args.p)
    else:
        dataset = None
        raise AssertionError(f"")

    print(dataset)
    train(model, dataset, train_args, tokenizer, sharding_size=args.shard_eval)