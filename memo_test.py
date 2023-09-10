import os
import argparse

from tqdm import tqdm
import torch
import torch.nn as nn
from data import get_dataset, get_data, CustomMLMCollator
import numpy as np
from transformers import AutoModelForMaskedLM, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default="bert-base-cased")

parser.add_argument('--data_type', default='huggingface')
parser.add_argument('--data', default='squad', required=False, help='default')
parser.add_argument('--use_partial_data', default=True, required=False)
parser.add_argument('--partial_data_size', default=8, type=int, required=False)
parser.add_argument('--split_test', default=0.2, type=float)

parser.add_argument('--seed', default=123, type=int)

# train
parser.add_argument('--lr', default=2e-5)
parser.add_argument('--wd', default=1e-2)
parser.add_argument('--epochs', default=30)
parser.add_argument('--b_train', default=8, type=int)
parser.add_argument('--max_steps', type=int, default=20000)
parser.add_argument('--chunk_size', default=64, type=int)

parser.add_argument('--p', default=.50, type=float)
parser.add_argument('--ada_mask', default=False, required=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--entropy', default=False, required=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--mrd', default=False, required=False, action=argparse.BooleanOptionalAction)

# Validation
parser.add_argument('--b_eval', default=8, type=int)
parser.add_argument('--shard_eval', default=300, type=int)

# Test
parser.add_argument('--test', default=False, required=False, action=argparse.BooleanOptionalAction)

# Log
parser.add_argument('--logging_steps', type=int, default=100)
parser.add_argument('--save_steps', type=int, default=20000)


def get_token_acc(_logits, _labels):
    preds = np.argmax(_logits, -1)

    preds = preds.reshape(-1)
    _labels = _labels.reshape(-1)

    mask = _labels != -100
    _labels = _labels[mask]
    preds = preds[mask]

    acc_token = accuracy_score(_labels, preds)

    return acc_token


def get_entropy(_logits, _labels):
    # Entropy
    mask = _labels != -100
    log_probs = np.log(np.exp(_logits) / np.exp(_logits).sum(axis=-1, keepdims=True))
    entropy = - np.sum(np.exp(log_probs) * log_probs, axis=-1)
    entropy = entropy[mask].mean(axis=-1)
    return entropy


def get_perr(_logits, _labels):
    preds = np.argmax(_logits, -1)
    mask = _labels != -100
    p_err = np.array(list(map(lambda p, l, m: ~ (p[m] == l[m]).all(), preds, _labels, mask))).mean()
    return p_err


def get_rankme(_repre, eps=1e-6):
    _, singular_vals, _ = np.linalg.svd(_repre)

    p = singular_vals / singular_vals.sum(axis=1, keepdims=True) + eps
    rankme = np.exp((- p * np.log(p)).sum(axis=1)).mean()
    return rankme


def get_memorization(_data: torch.Tensor, _model, mask_token_id):
    _inps = _data.clone()
    _batch_size = _inps.size(0)
    _chunk_size = _inps.size(1)

    mask = np.random.choice(np.arange(_chunk_size).astype(int), size=(_batch_size, ))
    mask_onehot = nn.functional.one_hot(torch.Tensor(mask).long().to('cuda:0'), num_classes=_chunk_size).bool()

    _labels = _inps.clone()
    _labels[~mask_onehot] = -100

    _inps[mask_onehot] = mask_token_id

    with torch.no_grad():
        _output = _model(_inps)
    _logits = _output['logits'].detach().cpu().numpy()
    _preds = _logits.argmax(-1)

    _labels = _labels.detach().cpu().numpy()
    mask_onehot = mask_onehot.detach().cpu().numpy()
    return np.mean(_preds[mask_onehot] == _labels[mask_onehot])


def evaluate(_model, _dataset):
    _model = _model.to('cuda:0')
    org_ids = _dataset['input_ids']
    _loss = 0
    _acc = 0
    _entropy = 0
    _rankme_emb = 0
    _rankme_repre = 0
    _cnt = 0
    _memo = 0

    _model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(org_ids), args.b_eval)):
            org_id = org_ids[i*args.b_eval:(i+1)*args.b_eval].to('cuda:0')
            if len(org_id) == 0:
                break
            _memo += get_memorization(org_id, _model, mlm_collator.tokenizer.convert_tokens_to_ids(mlm_collator.tokenizer.mask_token))

            # org_id, org_labels = mlm_collator.masking_tokens(org_id, args.p)
            #
            # org_output = _model(org_id, labels=org_labels, output_hidden_states=True)
            #
            # embs = org_output['hidden_states'][0].cpu().detach().numpy()
            # repre = org_output['hidden_states'][-1].cpu().detach().numpy()
            #
            # org_labels = org_labels.detach().cpu().numpy()
            # org_loss = org_output['loss'].detach().cpu().numpy()
            # org_logits = org_output['logits'].detach().cpu().numpy()
            #
            # _loss += org_loss.item()
            # _acc += get_token_acc(org_logits, org_labels)
            # _entropy += get_entropy(org_logits, org_labels)
            # _rankme_emb += get_rankme(embs)
            # _rankme_repre += get_rankme(repre)
            _cnt += 1

    _memo /= _cnt
    _loss /= _cnt
    _acc /= _cnt
    _entropy /= _cnt
    _rankme_emb /= _cnt
    _rankme_repre /= _cnt

    return _loss, _acc, _entropy, _rankme_emb, _rankme_repre, _memo


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

    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def compute_metrics(eval_preds, _model, eps=1e-6):
    # RankMe
    preds, labels, inps = eval_preds
    with torch.no_grad():
        _model.eval()
        embs = _model.bert.embeddings(torch.Tensor(inps).to('cuda:0').long())
        embs = embs.cpu().detach().numpy()





def print_fmt(_name, _val):
    print(f'{_name} {np.mean(_val)} +- {np.std(_val)} : {np.mean(_val): .2f}({np.std(_val): .2f})')


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CACHE_DIR'] = 'C:/Users/ay011/.cache/huggingface/datasets'
    os.environ['MASKING_P'] = str(args.p)

    os.environ['ITERATION_STEP'] = str(0)

    np.random.seed(args.seed)

    _model_path = args.ckpt if args.ckpt is not None else args.model_type
    model = AutoModelForMaskedLM.from_pretrained(_model_path)

    train_args = {'learning_rate': args.lr, 'num_train_epochs': args.epochs, 'weight_decay': args.wd,
                  'per_device_train_batch_size': args.b_train, 'per_device_eval_batch_size': args.b_eval,
                  'do_eval': True, 'do_train': True,
                  'max_steps': args.max_steps,
                  'logging_steps': args.logging_steps, 'save_steps': args.save_steps}

    dataset = get_dataset(args.data_type, args.data)
    dataset = dataset['train'].train_test_split(args.split_test)
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

    mlm_collator = CustomMLMCollator(tokenizer, args.p)

    print(dataset)
    print(f'ckpt : {args.ckpt}')
    losses = []
    acces = []
    entropies = []
    rankme_embs = []
    rankme_repreps = []
    memorizations = []
    for i in range(5):
        loss, acc, entropy, rankme_emb, rankme_repre, memorization = evaluate(model, dataset["test"])
        memorizations.append(memorization)
        losses.append(loss)
        acces.append(acc)
        entropies.append(entropy)
        rankme_embs.append(rankme_emb)
        rankme_repreps.append(rankme_repre)

    print_fmt('Memorization', memorizations)
    # print_fmt('ACC', acces)
    # print_fmt('Loss', losses)
    # print_fmt('Entropy', entropies)
    # print_fmt('RankMe:emb', rankme_embs)
    # print_fmt('RankMe:repre', rankme_repreps)

