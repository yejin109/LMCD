import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import pandas as pd
import torch
import argparse
import tqdm
import os
import glob

from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default="bert-base-cased")
parser.add_argument('--data_type', default='huggingface')
parser.add_argument('--data', default='bookcorpus')
parser.add_argument('--p', type=float, default=.2)
parser.add_argument('--max_size', type=int, default=1000000)
parser.add_argument('--fsize', type=int, default=15)


def get_data(dataset_name, tokenizer_name, maximum_size=None):
    cache_path = glob.glob(f'./cache/*_train_data.npy')
    cache_path = [i.replace('\\', '/') for i in cache_path]
    cache_path = {_path.split('/')[-1].split('_')[0]: _path for _path in cache_path}
    if dataset_name in cache_path.keys():
        with open(f'./cache/{dataset_name}_train_data.npy', 'rb') as f:
            _data = pickle.load(f)
    else:
        dataset = load_dataset(dataset_name, split='train')
        if maximum_size is not None:
            dataset = dataset.shard(math.floor(len(dataset)/maximum_size), 0)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenized_datasets = dataset.map(tokenize_function,
                                         batched=True, remove_columns=list(dataset.features.keys()),
                                         fn_kwargs={'_tokenizer': tokenizer})

        _data = []
        for data in tqdm.tqdm(tokenized_datasets['input_ids'], desc="Remove Special Token"):
            special_tk_mask = np.array(tokenizer.get_special_tokens_mask(data, already_has_special_tokens=True)).astype(
                bool)
            _data.append(np.array(data)[~special_tk_mask])

        with open(f'./cache/{dataset_name}_train_data.npy', 'wb') as f:
            pickle.dump(_data, f)

    return _data


def tokenize_function(examples, _tokenizer=None):
    result = _tokenizer(examples["text"])
    if _tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


def compute_lb(p, data):
    pyn = list(map(lambda x: len(set(np.array(x)[~ torch.bernoulli(torch.full(x.shape, p)).bool().cpu().numpy()])), data))
    pyn = np.array(pyn)

    zero_filter = pyn == 0

    pyn = pyn[~zero_filter]
    pyn = 1/pyn

    pxn = list(map(lambda x: len(set(x)), data))

    pxn = np.array(pxn)[~zero_filter]
    pxn = 1/np.array(pxn)

    ratio = np.mean(pxn/pyn)
    res = 1 - p**p*(1-p)**(1-p) * ratio
    print(f"{p} DONE")
    return res


if __name__ == '__main__':
    args = parser.parse_args()
    font_dict = {'fontsize': args.fsize}

    train_data = get_data(args.data, args.model_type, args.max_size)

    mask_p = np.arange(0.01, 0.99, step=0.01).tolist()

    print(compute_lb(.2, train_data))
    print(compute_lb(.4, train_data))
    print(compute_lb(.6, train_data))
    print(compute_lb(.8, train_data))

    # lower_bound = list(map(lambda x: compute_lb(x, train_data), mask_p))
    #
    # plt.figure()
    # plt.plot(mask_p, lower_bound, label='Lower Bound')
    # plt.xlabel('Masking Probability', fontdict=font_dict)
    # plt.ylabel('Perr', fontdict=font_dict)
    # plt.legend()
    # plt.savefig(f'{args.data}_lowerbound.png')
    # plt.show()
