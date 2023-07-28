import torch
import numpy as np
import argparse
from transformers import AutoTokenizer

from data import get_dataset
import pickle

from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default="bert-base-cased")
parser.add_argument('--data_type', default='huggingface')
parser.add_argument('--data', default='imdb')
parser.add_argument('--p', type=float, default=.4)

parser.add_argument('--chunk_size', type=int, default=128)
parser.add_argument('--window_size', type=int, default=33)
parser.add_argument('--search_iter', type=int, default=3)


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


def search_possible(_input_ids, masking_p, window_size, tk, output: defaultdict):
    dat = torch.Tensor(_input_ids)
    window_shape = (window_size, )

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
    # mask = np.random.binomial(n=1, p=masking_p, size=dat.shape[0]*dat.shape[1]).reshape(dat.shape)
    # skip_num = int(np.floor(window_size/2))
    skip_num = 4

    labels[mask == 0] = 0
    dat[mask == 1] = 0

    labels = labels.cpu().numpy().astype(int)
    dat = dat.cpu().numpy().astype(int)
    special_tokens_mask = special_tokens_mask.cpu().numpy()

    label_window = np.lib.stride_tricks.sliding_window_view(labels, window_shape, -1)[:, ::skip_num, :]
    label_window = label_window.reshape((-1, window_size))
    # label_window = labels

    data_window = np.lib.stride_tricks.sliding_window_view(dat, window_shape, -1)[:, ::skip_num, :]
    data_window = data_window.reshape((-1, window_size))
    # data_window = dat

    special_tokens_mask_window = np.lib.stride_tricks.sliding_window_view(special_tokens_mask, window_shape, -1)[:, ::skip_num, :]
    special_tokens_mask_window = special_tokens_mask_window.reshape((-1, window_size))

    print(f"Chunk num : {len(label_window)}")
    for data, label, special_token in tqdm(zip(data_window, label_window, special_tokens_mask_window)):
        if (label.sum() == 0) or (data.sum() == 0):
            continue
        data = data[~special_token]
        label = label[~special_token]
        output[tuple(sorted(np.unique(data[(data != 0)]).tolist()))].update(label[label != 0].tolist())
    return output


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = get_dataset(args.data_type, args.data)['train']

    tokenizer = AutoTokenizer.from_pretrained(args.model_type)
    tokenized_datasets = dataset.map(tokenize_function,
                                     batched=True, remove_columns=["text", "label"],
                                     fn_kwargs={'_tokenizer': tokenizer})

    chunk_size = args.chunk_size
    lm_datasets = tokenized_datasets.map(group_texts,
                                         batched=True, remove_columns=["token_type_ids", 'attention_mask', 'word_ids'],
                                         fn_kwargs={'_chunk_size': chunk_size})
    # input_ids : (Chunk 개수, chunk size) = (D, 256)

    masked_data = defaultdict(set)
    for i in range(args.search_iter):
        print(f"{i+1}th started")
        masked_data = search_possible(lm_datasets['input_ids'],
                                      masking_p=args.p,
                                      window_size=args.window_size,
                                      tk=tokenizer,
                                      output=masked_data)
        print(f"Key num : {len(masked_data)}")
    with open(f'./results/{args.data}_window{args.window_size}_chunk{args.chunk_size}_p{int(args.p*100)}.pkl', 'wb') as f:
        pickle.dump(masked_data, f)
