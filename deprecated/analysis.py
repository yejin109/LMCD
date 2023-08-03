import pickle
import glob
import tqdm
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.metrics import mutual_info_score


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default="bert-base-cased")
parser.add_argument('--data_type', default='huggingface')
parser.add_argument('--data', default='imdb')
parser.add_argument('--p', type=float, default=.4)

parser.add_argument('--window_size', type=int, default=33)
parser.add_argument('--chunk_size', type=int, default=128)


def load_mtoken_dict(_dir):
    with open(_dir, 'rb') as f:
        _dat = pickle.load(f)
    f.close()
    return _dat


def tokenize_function(examples, _tokenizer=None):
    result = _tokenizer(examples["text"])
    if _tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


def group_texts(examples, _chunk_size=None):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // _chunk_size) * _chunk_size

    # Split by chunks of max_len
    result = {
        k: [t[i : i + _chunk_size] for i in range(0, total_length, _chunk_size)]
        for k, t in concatenated_examples.items()
    }

    # Create a new labels column
    # result["labels"] = result["input_ids"].copy()
    return result


def masking(_input_ids, masking_p, tk):
    if not isinstance(_input_ids, np.ndarray):
        dat = np.array(_input_ids)
    else:
        dat = _input_ids.astype(dtype=np.uint32)
    dat = torch.Tensor(_input_ids)

    labels = dat.clone()
    # Masking 한 것에서 보는 걸로
    probability_matrix = torch.full(labels.shape, masking_p)
    special_tokens_mask = [
        tk.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()

    mask = masked_indices.cpu().numpy()
    labels[mask == 0] = -100
    dat[mask == 1] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token) if tokenizer is not None else 0

    labels = labels.long().cpu().numpy()
    dat = dat.long().cpu().numpy()

    return dat, labels


def possible_max_cnt(_dat, _label, _window_size, tk_dict, special_tk):
    mask_token_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token) if tokenizer is not None else 0
    masked_ids = np.argwhere(_label != mask_token_id).squeeze(-1)
    width = int(np.floor(_window_size/2))
    res = -1
    nonzero_cnt = 0
    _dat = _dat[~special_tk]
    for masked_id in masked_ids:
        _seq = _dat[max(masked_id-width, 0):min(masked_id+width+1, _dat.shape[0])]
        _seq = sorted(np.unique(_seq[_seq != 0]))
        cnt = len(tk_dict[tuple(_seq)])
        # cnt = 0
        # while len(_seq) != 0:
        #     _cnt = len(tk_dict[tuple(_seq)])
        #     if _cnt != 0:
        #         # cnt = _cnt if _label[masked_id] in tk_dict[tuple(_seq)] else 0
        #         cnt = _cnt if _label[masked_id] in tk_dict[tuple(_seq)] else 0
        #         break
        #     _seq.pop(-1)
        if cnt != 0:
            nonzero_cnt += 1
        if cnt > res:
            res = cnt
    _ratio = nonzero_cnt / len(masked_ids)
    return res, _ratio


if __name__ == '__main__':
    args = parser.parse_args()

    mtoken_dict = load_mtoken_dict(f'./results/{args.data}_window{args.window_size}_chunk{args.chunk_size}_p{int(args.p*100)}.pkl')
    # mtoken_dict = load_mtoken_dict(f'./results/{args.data}_window33_chunk{args.chunk_size}_p{int(args.p*100)}.pkl')

    dataset = load_dataset(args.data)['train']

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    tokenized_datasets = dataset.map(tokenize_function,
                                     batched=True, remove_columns=["text", "label"],
                                     fn_kwargs={'_tokenizer': tokenizer})

    lm_datasets = tokenized_datasets.map(group_texts,
                                         batched=True, remove_columns=["token_type_ids", 'attention_mask', 'word_ids'],
                                         fn_kwargs={'_chunk_size': args.chunk_size})

    inputs, labels = masking(lm_datasets['input_ids'], args.p, tokenizer)
    lm_datasets = None
    tokenized_datasets = None
    dataset = None
    print('Message num : ', len(inputs))

    max_cnts = - np.ones(inputs.shape[0])
    accept_ratios = 0
    mtoken_key_list = list(mtoken_dict.keys())
    nums = 0
    special_tokens_mask = np.array([
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in inputs.tolist()
    ], dtype=bool)
    for i, (input_i, label_i, special_token_i) in tqdm.tqdm(enumerate(zip(inputs, labels, special_tokens_mask))):
        max_cnt, _accept_ratio = possible_max_cnt(input_i, label_i, args.window_size, mtoken_dict, special_token_i)
        if max_cnt == 0:
            continue
        max_cnts[i] = max_cnt
        # max_cnts.append(max_cnt)
        accept_ratios += _accept_ratio
        nums += 1
    # max_cnts = np.array(max_cnts)
    max_cnts = max_cnts[max_cnts != -1]
    print(np.mean(1 - 1 / max_cnts), accept_ratios/nums)

    # plt.figure()
    # sns.histplot(data, x="log(duplicates)", hue="p", element="step", stat="density", common_norm=False,)
    # plt.show()
    print()
