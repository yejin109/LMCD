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
parser.add_argument('--p', type=float, default=.2)

parser.add_argument('--chunk_size', type=int, default=128)
parser.add_argument('--window_size', type=int, default=5)


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


def search_possible(_input_ids, masking_p, window_size):
    if not isinstance(_input_ids, np.ndarray):
        dat = np.array(_input_ids)
    else:
        dat = _input_ids.astype(dtype=np.uint32)

    window_shape = (window_size, 1)

    # Masking 한 것에서 보는 걸로
    mask = np.random.binomial(n=1, p=masking_p, size=dat.shape[0]*dat.shape[1]).reshape(dat.shape)
    labels = dat.copy()
    labels[mask == 0] = 0
    dat[mask == 1] = 0

    label_window = np.lib.stride_tricks.sliding_window_view(labels, window_shape).squeeze(-1)
    label_window = label_window.reshape((-1, window_size))

    data_window = np.lib.stride_tricks.sliding_window_view(dat, window_shape).squeeze(-1)
    data_window = data_window.reshape((-1, window_size))

    res = defaultdict(list)
    print(f"Chunk num : {len(label_window)}")
    for data, label in tqdm(zip(data_window, label_window)):
        if (label.sum() == 0) or (data.sum() == 0):
            continue
        res[tuple(data[data != 0])].extend(label[label != 0].tolist())
    return res


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

    masked_data = search_possible(lm_datasets['input_ids'], masking_p=args.p, window_size=args.window_size)
    dist = list(map(lambda x: len(x), masked_data.values()))

    with open(f'./results/{args.data}_chunk{args.chunk_size}_p{int(args.p*100)}.pkl', 'wb') as f:
        pickle.dump(masked_data, f)
