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
from scipy.stats.mstats import gmean
from itertools import combinations

from collections import defaultdict
from datasets import load_dataset
from transformers import AutoTokenizer


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default="bert-base-cased")
parser.add_argument('--data_type', default='huggingface')

# Data
parser.add_argument('--data', default='bookcorpus')
# parser.add_argument('--data', default='imdb')
parser.add_argument('--dataset_max', type=int, default=1000000)

# Experiment
parser.add_argument('--p', type=float, default=.2)
parser.add_argument('--n', type=int, default=64)

# Visualization
parser.add_argument('--fsize', type=int, default=15)


def get_data(dataset_name, tokenizer_name, chunk_size, maximum_size=None):
    cache_path = glob.glob(f'./cache/*_train_data_*.npy')
    cache_path = [i.replace('\\', '/') for i in cache_path]
    cache_path = {_path.split('/')[-1]: _path for _path in cache_path}
    fname = f'{dataset_name}_train_data_{maximum_size}.npy'
    if fname in cache_path.keys():
        with open(f'./cache/{fname}', 'rb') as f:
            _data = pickle.load(f)
    else:
        dataset = load_dataset(dataset_name, split='train')
        if maximum_size is not None:
            dataset = dataset.shard(math.ceil(len(dataset)/maximum_size), 0)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenized_datasets = dataset.map(tokenize_function,
                                         batched=True, remove_columns=list(dataset.features.keys()),
                                         fn_kwargs={'_tokenizer': tokenizer})

        tokenized_datasets = tokenized_datasets.map(group_texts,
                                                    batched=True,
                                                    fn_kwargs={'_chunk_size': chunk_size})

        _data = []
        for data in tqdm.tqdm(tokenized_datasets['input_ids'], desc="Remove Special Token"):
            special_tk_mask = np.array(tokenizer.get_special_tokens_mask(data, already_has_special_tokens=True)).astype(
                bool)
            _data.append(np.array(data)[~special_tk_mask])

        with open(f'./cache/{dataset_name}_train_data_{maximum_size}.npy', 'wb') as f:
            pickle.dump(_data, f)

    return _data


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
        k: [t[i: i + _chunk_size] for i in range(0, total_length, _chunk_size)]
        for k, t in concatenated_examples.items()
    }
    return result


def compute_lb(p, sample_size, method='org', verbose=False):
    _size = sample_size
    _p = p
    mask = torch.bernoulli(torch.full((_size, ), _p)).bool().cpu().numpy()
    res = np.ones(_size)
    res[mask] = _p
    res[~mask] = 1-_p
    return 1 - np.mean(res)
    # dmc_factor = p**p*(1-p)**(1-p)
    # res = 1 - dmc_factor
    # if verbose:
    #     log = [f'{p} DONE']
    #     log.append(f'Res {res:.4f}')
    #     log.append(f'DMC {dmc_factor:.4f}')
    #     print("\n\t".join(log))
    #     print()
    # return res
    # dist = defaultdict(int)
    # dist.update(_dist)

    # def estimate(_seq, _mask: np.ndarray):
    #     if _mask.sum() == 0:
    #         return 1
    #     if _mask[0]:
    #         if p > 0.5:
    #             return (1-p+dist[_seq[0]])/2
    #         else:
    #             return 1
    #         # return dist[_seq[0]]
    #     else:
    #         return 1
        # if p > 0.5:
        #     return 1
        # else:
        #     return 0

        # _mask_val = [dist[x] for x in _seq[_mask]]
        # return (1-p)**(1-p)
        # return np.max(_mask_val) / p
        # if _mask[0]:
        #     return dist[_seq[0]]
        #     # return dist[_seq[np.argwhere(_mask).flatten()[0]]] / p
        # else:
        #     return 1
        #     # return dist[_seq[np.argwhere(_mask).flatten()[0]]] / ((1-p)*dist[_seq[0]])

        #
        # # return min(dist[_seq[np.argwhere(_mask).flatten()[0]]] * 100, 0.5)
        # return (1 - p)**(0.5)

    # mask = list(map(lambda x: ~ torch.bernoulli(torch.full(x.shape, p)).bool().cpu().numpy(), data))
    # mask_dist = list(map(lambda x: (np.sum(x), x.shape[0]-np.sum(x)), mask))
    # mask_dist = np.array(mask_dist)

    # # Past
    # cnt_yn = list(map(lambda x, m: len(set(np.array(x)[m])), data, mask))
    # cnt_yn = np.array(cnt_yn)
    #
    # zero_filter = cnt_yn == 0
    #
    # cnt_xn = list(map(lambda x: len(set(x)), data))
    # cnt_xn = np.array(cnt_xn)
    #
    # cnt_yn = cnt_yn[~zero_filter]
    # cnt_xn = cnt_xn[~zero_filter]
    #
    # prob_yn = 1/cnt_yn
    # prob_xn = 1/cnt_xn
    #
    # ratio = np.mean(prob_xn/prob_yn)
    # res = 1 - p**p*(1-p)**(1-p) * ratio
    #
    # log = [f'{p} DONE']
    # log.append(f'Res {res:.4f}')
    # log.append(f'DMC {p*(1-p)**(1-p):.4f}')
    # log.append(f'ratio {np.mean(ratio):.4f}')
    # print("\n\t".join(log))

    # a = -1
    # b = 1
    # res_p = 1 - p**p*(1-p)**(1-p) * p**a * (1-p)**b
    # log = [f'{p} DONE']
    # log.append(f'Ratio {ratio}')
    # log.append(f'Res {res}')
    # log.append(f'Res p {res_p}')
    # print("\n\t".join(log))

    # Ver 2
    # zero_filter = np.array(list(map(lambda x: x[1] != 0, mask_dist)))
    # masked_token = list(map(lambda x, m: x[~m], data, mask))
    # masked_token = [m_token for i, m_token in enumerate(masked_token) if zero_filter[i]]
    #
    # masked_token_prob_max = list(map(lambda x: np.array(list(map(lambda y: dist[y], x))).mean(), masked_token))
    # masked_token_prob_max = np.array(masked_token_prob_max)
    #
    # # actual_n = mask_dist.sum(axis=1, keepdims=True)
    # # actual_n = actual_n * np.array([1 - p, p])[np.newaxis, :]
    # # actual_n = mask_dist - actual_n
    # # actual_n = (actual_n > 0).astype(float)
    #
    # actual_n = mask_dist.sum(axis=1)
    # actual_n = mask_dist[:, 0] / actual_n
    #
    # actual_n = actual_n[zero_filter]
    # # dmc_factor = np.prod(np.tile(np.array([1-p, p])[np.newaxis, :], (len(actual_n), 1)) ** actual_n, axis=1)
    #
    # # res_max = 1 - np.mean(masked_token_prob_max * dmc_factor)
    # # res_max = 1 - np.mean(dmc_factor)
    # # res = res_max
    #
    # res = 1 - p**p*(1-p)**(1-p) * np.mean(actual_n)
    #
    # # masked_token = list(map(lambda x, m: x[~m], data, mask))
    # # masked_token_prob_log = np.array(list(map(lambda x: np.array(list(map(lambda y: np.log(dist[y]), x))).sum(), masked_token)))
    # #
    # # actual_mask_prob_log = np.tile(np.array([np.log(p), np.log(1-p)])[np.newaxis, :], (len(data), 1)) * mask_dist
    # # actual_mask_prob_log = actual_mask_prob_log.sum(axis=1)
    # #
    # # masked_token_prob_log = masked_token_prob_log[zero_filter]
    # # actual_mask_prob_log = actual_mask_prob_log[zero_filter]
    # # ratio = masked_token_prob_log - actual_mask_prob_log
    # #
    # # dmc_factor = (p ** p * (1 - p) ** (1 - p)) ** n
    # # n = mask_dist.sum(axis=1)
    # # n = n[zero_filter]
    # # dmc_factor = p**p*(1-p)**(1-p)
    # #
    # # res = dmc_factor + ratio
    # # res = np.mean(1 - np.exp(res))
    # log = [f'{p} DONE']
    # # log.append(f'Prob max {masked_token_prob_max}')
    # log.append(f'Res {res:.4f}')
    # print("\n\t".join(log))

    # # Ver 3 : Unigram assumption -> minimum token prob
    # dmc_factor = (p ** p * (1 - p) ** (1 - p)) ** n
    # zero_filter = np.array(list(map(lambda x: x[1] != 0, mask_dist)))
    #
    # masked_token = list(map(lambda x, m: x[~m], data, mask))
    # masked_token = [m_token for i, m_token in enumerate(masked_token) if zero_filter[i]]
    #
    # masked_token_prob_max = list(map(lambda x: np.array(list(map(lambda y: dist[y], x))).max(), masked_token))
    # res_max = 1 - np.mean(masked_token_prob_max)
    #
    # masked_token_prob_first = list(map(lambda x: dist[x[0]], masked_token))
    # res_first = 1 - np.mean(masked_token_prob_first)
    #
    # # masked_token_prob_gmean = list(map(lambda x: gmean(np.array(list(map(lambda y: dist[y], x)))), masked_token))
    # # res_gmean = 1 - np.mean(masked_token_prob_gmean)
    # log = [f'{p} DONE']
    # log.append(f'Unigram {res_max}')
    # log.append(f'First {res_first}')
    # print("\n\t".join(log))
    #
    # res = res_first

    # Ver 4
    # dmc_factor = (p ** p * (1 - p) ** (1 - p))
    # # masked = torch.bernoulli(torch.full((len(data),), p)).bool().cpu().numpy()
    # mask = list(map(lambda x: torch.bernoulli(torch.full(x.shape, p)).bool().cpu().numpy(), data))
    #
    # p_val = min(1/(1-p), 1/dmc_factor)
    # # masked_val = list(map(lambda x: np.array(list(map(lambda y: dist[y], x))) / p, data))
    # # masked_val = list(map(lambda x: np.array(list(map(lambda y: dist[y], x))) * np.clip(1/p, 0, 1/(1-p)), data))
    # # masked_val = list(map(lambda x: np.array(list(map(lambda y: dist[y], x))) * min(1/p, 1/dmc_factor), data))
    #
    # # ratio = list(map(lambda m, m_val: m.astype(int) * m_val + (~m).astype(int) * np.clip(1/(1-p), 0, 1/p), mask, masked_val))
    # # ratio = [m.astype(int) * m_val + (~m).astype(int) * 1 / (1 - p) for m, m_val in zip(mask, masked_val)]
    # # ratio = [m.astype(int) * m_val + (~m).astype(int) * min(1/(1-p), 1/dmc_factor) for m, m_val in zip(mask, masked_val)]
    #
    # if method == 'org':
    #     ratio = list(map(lambda x, m: estimate(x, m), data, mask))
    #
    #     # masked_val = list(map(lambda x: np.array(list(map(lambda y: dist[y], x))) / p, data))
    #     # masked_val = np.array(list(map(lambda x: dist[x[0]]/p, data)))
    #     # ratio = [m.astype(int) * m_val + (~m).astype(int) * 1 / (1 - p) for m, m_val in zip(mask, masked_val)]
    #     # ratio = list(map(lambda x: x[0], ratio))
    #     # masked_val = np.array(list(map(lambda x: dist[x[0]]/p, data)))
    #     # masked_val = list(map(lambda x: np.array(list(map(lambda y: dist[y], x)))/p, data))
    #     # ratio = [np.prod(m.astype(int) * m_val + (~m).astype(int) * 1/(1-p)) for m, m_val in zip(mask, masked_val)]
    #     # ratio = np.concatenate((np.ones((len(data), 1))*(1/(1-p)), ratio), axis=1)
    #     # ratio = np.array([ratio[i, int(m)] for i, m in enumerate(mask)])
    #     # ratio = [m.astype(int) * m_val + (~m).astype(int) * 1 / (1 - p) for m, m_val in zip(mask, masked_val)]
    # elif method == 'clip-half':
    #     ratio = np.array(list(map(lambda x: dist[x[0]]/p, data)))[:, np.newaxis]
    #     ratio = np.concatenate((np.clip(np.ones((len(data), 1))*(1/(1-p)), 0, 1/p), ratio), axis=1)
    # elif method == 'clip-all':
    #     masked_val = np.array(list(map(lambda x: dist[x[0]] * np.clip(1/p, 0, 1/(1-p)), data)))
    #     # masked_val = list(map(lambda x: np.array(list(map(lambda y: dist[y], x))) * np.clip(1/p, 0, 1/(1-p)), data))
    #     ratio = list(map(lambda m, m_val: m.astype(int) * m_val + (~m).astype(int) * np.clip(1/(1-p), 0, 1/p), mask, masked_val))
    #     # ratio = list(map(lambda x: x[0], ratio))
    #     # ratio = list(map(lambda x: np.prod(x), ratio))
    #     # ratio = list(map(lambda x, m: np.prod(x[:math.ceil(len(mask)*p)]), ratio, mask))
    #     # masked_val = list(map(lambda x: np.array(list(map(lambda y: dist[y], x)))/p, data))
    #     # ratio = [np.prod(m.astype(int) * m_val + (~m).astype(int) * 1/(1-p)) for m, m_val in zip(mask, masked_val)]
    #
    #     # ratio = np.array(list(map(lambda x: dist[x[0]], data)))[:, np.newaxis] * p_val
    #     # ratio = np.array(list(map(lambda x: np.array(list(map(lambda y: dist[y], x))), data)))[:, np.newaxis]
    #     # # ratio = ratio * np.clip(np.ones((len(ratio), 1))*(1/p), 0, 1/(1-p))
    #     # # ratio = np.concatenate((np.clip(np.ones((len(data), 1)) * (1 / (1 - p)), 0, 1 / p), ratio), axis=1)
    #     # ratio = np.concatenate((np.ones((len(data), 1))*p_val, ratio), axis=1)
    #     # ratio = np.array([ratio[i, int(m)] for i, m in enumerate(masked)])
    # else:
    #     ratio = None
    # ratio = np.array(ratio)
    # ratio = ratio.mean()
    #
    # # ratio = 1
    # # ratio = max(1, ratio)
    # # ratio = np.clip(ratio, a_min=1, a_max=None)
    #
    # res = 1 - dmc_factor * ratio
    # # res = 1 / (1 + dmc_factor)
    #
    # # res = min(res, dmc_factor)
    # # ratio = np.array(list(map(lambda x: dist[x[0]]/p, data)))[:, np.newaxis]
    # # # ratio = np.concatenate((np.ones((len(data), 1))*(1/(1-p)), ratio), axis=1)
    # # ratio = np.concatenate((np.clip(np.ones((len(data), 1))*(1/(1-p)), 0, 1/p), ratio), axis=1)
    # # ratio = np.array([ratio[i, int(m)] for i, m in enumerate(masked)])
    # # res = 1 - np.mean(dmc_factor * ratio)
    #
    # log = [f'{p} DONE']
    # log.append(f'Res {res:.4f}')
    # log.append(f'DMC {dmc_factor:.4f}')
    # log.append(f'ratio {np.mean(ratio):.4f}')
    # print("\n\t".join(log))
    # return res


def get_exp_data(_ps):
    def update_name(name):
        if 'bert-large-uncased' in name:
            return 'bert-large-uncased'
        elif 'bert-base-cased' in name:
            return 'bert-base-cased'
        elif 'medium' in name:
            return 'prajjwal1/bert-medium'
        elif 'mini' in name:
            return 'prajjwal1/bert-mini'
        elif 'tiny' in name:
            return 'prajjwal1/bert-tiny'

    _data = {}
    for p in _ps:
        dat = pd.read_csv(f'./results/{args.data}_{p}.csv', index_col=0)
        cols = list(filter(lambda x: ('MIN' not in x) and ('MAX' not in x), [col for col in dat.columns]))
        dat = dat.loc[:, cols]

        cols = [update_name(col) for col in dat.columns]
        dat.columns = cols

        dat = dat.iloc[-10:].mean()
        _data[p] = dat
    return _data


def visualize(_ps, _data):
    def get_size(name):
        def get_scaled(val):
            # return np.log(10*val) + 5
            return val
        if 'large' in name:
            return get_scaled(340)
        elif 'base' in name:
            return get_scaled(110)
        elif 'medium' in name:
            return get_scaled(41.7)
        elif 'mini' in name:
            return get_scaled(11.3)
        elif 'tiny' in name:
            return get_scaled(4.4)

    def get_color(name):
        if 'bert-large-uncased' in name:
            return 'blue'
        elif 'base' in name:
            return 'red'
        elif 'medium' in name:
            return 'green'
        elif 'mini' in name:
            return 'yellow'
        elif 'tiny' in name:
            return 'purple'

    points = defaultdict(list)
    for p in _ps:
        _mask_p = float(p[1:])/100
        for idx in _data[p].index:
            points[idx].append(_data[p][idx])

    plt.figure(figsize=(7, 7))
    plt.plot(mask_p, lower_bound, label='Lower Bound')
    # plt.plot(mask_p, lower_bound_clip, label='Lower Bound Clip')
    for idx, vals in points.items():
        plt.scatter([float(p[1:]) / 100 for p in _ps], vals, label=idx, c=get_color(idx), s=get_size(idx), alpha=0.5)
    plt.xlabel('Masking Probability', fontdict=font_dict)
    plt.ylabel('Perr', fontdict=font_dict)
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.legend()
    plt.title(args.data, fontdict=font_dict)
    plt.tight_layout()
    plt.savefig(f'{args.data}_lowerbound.png')
    plt.show()


def get_dist(_data: list):
    cnt_map = defaultdict(int)

    sample_idx = np.random.randint(len(_data), size=1000)
    norm = len(_data)
    for i, line in tqdm.tqdm(enumerate(_data), desc='Distribution count'):
        # cnt_map[line[0]] += 1
        # norm += 1
        for token in list(set(line)):
            cnt_map[token] += 1


        # for token in combinations(line, 2):
        #     cnt_map[tuple(sorted(token))] += 1
        #     norm += 1

    cnt_map = {k: v/norm for k, v in cnt_map.items()}
    return cnt_map


def generate_perf_tab(_ps, _data, _lower_bound):
    def get_p_idx(target_p):
        return np.argmin(np.abs(np.arange(0.01, 0.99, step=0.01)-target_p))

    def bold_char(char):
        return "\\textbf{" + char + "}"

    def get_line(model_name):
        _res = [bold_char(model_name)] + lines[name_to_key[model_name]]
        return ' & '.join(_res) + " \\\\ \midrule"

    name_to_key = {
        "BERT-tiny": "prajjwal1/bert-tiny",
        "BERT-mini": 'prajjwal1/bert-mini',
        "BERT-medium": 'prajjwal1/bert-medium',
        "BERT-base": 'bert-base-cased',
        "BERT-large": 'bert-large-uncased',
        "Lower Bound": "Lower Bound"
    }

    prefix = [
        "\\begin{table}[hbt!]",
        "\\begin{tabular}{@{}ccccc@{}}",
        "\\toprule"
    ]
    suffix = [
        "\end{tabular}",
        "\end{table}"
    ]

    lines = defaultdict(list)
    body = ["Masking Probability"]
    for p in _ps:
        _mask_p = float(p[1:])/100
        for idx in _data[p].index:
            lines[idx].append(f"{_data[p][idx]:.4f}")
        lines['Lower Bound'].append(f"{_lower_bound[get_p_idx(_mask_p)]: .4f}")
        body.append(str(_mask_p))
    body = [" & ".join(body)+"\\\\ \midrule"]

    order = ["BERT-tiny", "BERT-mini", "BERT-medium", "BERT-base", "BERT-large", "Lower Bound"]
    body = body + [get_line(m) for m in order]
    res = prefix + body + suffix
    res = "\n".join(res)
    return res


def generate_metric_tab(_data, _lower_bound, _masking_p):
    def get_p_idx(_target_p):
        return np.argmin(np.abs(_masking_p - _target_p))

    def bold_char(char):
        return "\\textbf{" + char + "}"

    def get_line(model_name):
        _res = [bold_char(model_name)] + lines[name_to_key[model_name]]
        return ' & '.join(_res) + " \\\\ \midrule"

    name_to_key = {
        "BERT-tiny": "prajjwal1/bert-tiny",
        "BERT-mini": 'prajjwal1/bert-mini',
        "BERT-medium": 'prajjwal1/bert-medium',
        "BERT-base": 'bert-base-cased',
        "BERT-large": 'bert-large-uncased',
        "Lower Bound": "Lower Bound"
    }

    prefix = [
        "\\begin{table}[hbt!]",
        "\\begin{tabular}{@{}ccccc@{}}",
        "\\toprule"
    ]
    suffix = [
        "\end{tabular}",
        "\end{table}"
    ]

    lines = defaultdict(list)
    body = ["Metric"]
    for p in _data.keys():
        _mask_p = float(p[1:])/100
        for idx in _data[p].index:
            lines[idx].append(f"{(_data[p][idx]-_lower_bound[get_p_idx(_mask_p)])/(1-_lower_bound[get_p_idx(_mask_p)])*100:.4f}")
        body.append(str(_mask_p))
    body = [" & ".join(body) + "\\\\ \midrule"]

    order = ["BERT-tiny", "BERT-mini", "BERT-medium", "BERT-base", "BERT-large", "Lower Bound"]
    body = body + [get_line(m) for m in order]
    res = prefix + body + suffix
    res = "\n".join(res)

    return res


def compute_ub(_p, _size, vocab_size):
    # mask = torch.bernoulli(torch.full((_size, ), _p)).bool().cpu().numpy()
    # _res = mask.astype(float)
    # _res = _p ** _res * (1-_p)**(1-_res) * vocab_size**(_res -1 + _p)
    # _res_log = np.log(_res) * 64
    # _res = np.sum(_res_log)

    _res = vocab_size**(_p) * _p ** _p * (1-_p) ** (1-_p)
    _res = np.log(_res)
    _res = _res * 1
    return np.exp(_res)


if __name__ == '__main__':
    args = parser.parse_args()
    font_dict = {'fontsize': args.fsize}

    train_data = get_data(args.data, args.model_type, chunk_size=args.n, maximum_size=args.dataset_max)

    token_dist = get_dist(train_data)

    # Lower Bound
    # # compute_lb(.2, args.n, token_dist, train_data, verbose=True)
    # # compute_lb(.4, args.n, token_dist, train_data, verbose=True)
    # # compute_lb(.6, args.n, token_dist, train_data, verbose=True)
    # # compute_lb(.8, args.n, token_dist, train_data, verbose=True)
    #
    # mask_p = np.arange(0.01, 0.99, step=0.01)
    # # mask_p = np.arange(0.05, 0.95, step=0.05).tolist()
    # lower_bound = []
    # for i in tqdm.tqdm(range(100)):
    #     lbs = list(map(lambda x: compute_lb(x, len(train_data)), mask_p))
    #     lower_bound.append(lbs)
    # lower_bound = np.array(lower_bound).mean(axis=0)
    # # lower_bound = list(map(lambda x: compute_lb(x, token_dist, train_data), mask_p))
    # # lower_bound_clip = list(map(lambda x: compute_lb(x, token_dist, train_data, method='clip-all'), mask_p))
    # # lower_bound_half = list(map(lambda x: compute_lb(x, token_dist, train_data, method='clip-half'), mask_p))
    #
    # ps = ['p20', 'p40', 'p60', 'p80']
    # data = get_exp_data(ps)
    # # visualize(ps, data)
    #
    # print("Metric")
    # metric_tab = generate_metric_tab(data, lower_bound, mask_p)
    # print(metric_tab)
    # print("Performance")
    # latex_table = generate_perf_tab(ps, data, lower_bound)
    # print(latex_table)


    # Upper bound
    n = 128
    print(f"Cardinality of vocabulary : {len(token_dist.keys())}")
    mask_p = np.arange(0.01, 0.99, step=0.01)

    # mask_p = np.repeat(mask_p, n, axis=1)

    upper_bounds = list(map(lambda p: compute_ub(p, 1000, len(token_dist.keys())), mask_p))
    # card = np.ones_like(mask_p) * len(token_dist.keys())
    # n_mat = np.arange(1, n+1, step=1)[np.newaxis, :]
    # n_mat = np.repeat(n_mat, mask_p.shape[0], axis=0)
    #
    # res = (mask_p ** mask_p * (1-mask_p)**(1-mask_p))**n_mat * card

    # card = np.log(card)
    #
    # res_log = mask_p * card + n_mat * mask_p * np.log(mask_p) + n_mat * (1-mask_p) * np.log(1-mask_p)
    # res = np.exp(res_log)
    for p, ub in zip(mask_p, upper_bounds):
        print(p, ub)
    print()


