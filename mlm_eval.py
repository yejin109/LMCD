import datetime
import os
import argparse

import torch
from data import get_dataset, get_data, CustomMLMCollator
from _utils import CustomWandbCallback, MaskingCallback
import numpy as np
from transformers import AutoModelForMaskedLM, TrainingArguments, AutoTokenizer
from sklearn.metrics import accuracy_score
from huggingface import CustomTrainer


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default="prajjwal1/bert-medium")
parser.add_argument('--ckpt', default=None)

parser.add_argument('--data_type', default='huggingface')
parser.add_argument('--data', default='bookcorpus', required=False, help='default bookcorpus, wikipedia')
parser.add_argument('--use_partial_data', default=True, required=False)
parser.add_argument('--partial_data_size', default=8, type=int, required=False)

parser.add_argument('--seed', default=123, type=int)

# train
parser.add_argument('--lr', default=2e-5)
parser.add_argument('--wd', default=1e-2)
parser.add_argument('--epochs', default=30)
parser.add_argument('--b_train', default=8, type=int)
parser.add_argument('--max_steps', type=int, default=20000)
parser.add_argument('--chunk_size', default=64, type=int)

parser.add_argument('--p', default=.20, type=float)
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


def get_token_acc(preds, labels):
    preds = np.argmax(preds, -1)

    preds = preds.reshape(-1)
    labels = labels.reshape(-1)

    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    acc_token = accuracy_score(labels, preds)

    return acc_token


def evaluate(_model, _dataset):
    org_ids = _dataset['input_ids']
    with torch.no_grad():
        _model.eval()

        org_ids, org_labels = mlm_collator.masking_tokens(torch.Tensor(org_ids).to('cuda:0').long(), args.p)
        org_output = _model(org_ids, org_labels)
        org_loss = org_output['loss'].detach().cpu().numpy()
        org_logits = org_output['logits'].detach().cpu().numpy()
        org_acc = get_token_acc(org_logits, org_labels.detach().cpu().numpy())
        org_labels = None
        org_logits = None
        org_ids = None

    return org_acc, org_loss


def train(_model, _dataset, _train_args, _tk, sharding_size=600):
    training_args = TrainingArguments(
        output_dir=os.environ['LOG_DIR'],
        evaluation_strategy="steps",  # candidates : steps, epochs
        include_inputs_for_metrics=True,
        **_train_args,
    )

    mlm_collator = CustomMLMCollator(_tk, args.p)

    trainer = CustomTrainer(
        model=_model,
        args=training_args,
        train_dataset=_dataset["train"],
        eval_dataset=_dataset["test"].shard(num_shards=sharding_size, index=np.random.randint(0, sharding_size, size=1)),
        data_collator=mlm_collator,
        compute_metrics=compute_metrics,
    )

    eval_res = trainer.evaluate()
    print(f"Evaludation : {eval_res['eval_loss']}")


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


def write_env_var(_name, val):
    if _name not in os.environ.keys():
        os.environ[_name] = str(val)


def compute_metrics(eval_preds, _model, eps=1e-6):
    write_env_var('EVAL_CNT', str(0))
    # Entropy
    preds, labels, inps = eval_preds
    mask = labels != -100
    log_probs = np.log(np.exp(preds) / np.exp(preds).sum(axis=-1, keepdims=True))
    entropy = - np.sum(np.exp(log_probs) * log_probs, axis=-1)
    entropy = entropy[mask].mean(axis=-1)

    # RankMe
    preds, labels, inps = eval_preds
    with torch.no_grad():
        _model.eval()
        embs = _model.bert.embeddings(torch.Tensor(inps).to('cuda:0').long())
        embs = embs.cpu().detach().numpy()
    _, singular_vals, _ = np.linalg.svd(embs)

    p = singular_vals / singular_vals.sum(axis=1, keepdims=True) + eps
    rankme = np.exp((- p * np.log(p)).sum(axis=1)).mean()

    # P_err
    preds, labels, inps = eval_preds
    preds = np.argmax(preds, -1)
    mask = labels != -100
    p_err = np.array(list(map(lambda p, l, m: ~ (p[m] == l[m]).all(), preds, labels, mask))).mean()

    # Token_acc
    preds, labels, inps = eval_preds
    preds = np.argmax(preds, -1)

    preds = preds.reshape(-1)
    labels = labels.reshape(-1)

    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    acc_token = accuracy_score(labels, preds)

    # Metric
    _p = float(os.environ['MASKING_P'])
    metric_cur = acc_token / (1 - _p)
    write_env_var('P_METRIC', str(metric_cur))
    write_env_var('ENTROPY', str(entropy))
    write_env_var('P_CNT', str(0))

    if (args.ada_mask or args.mrd) and int(os.environ['EVAL_CNT']) > 10:
        _increment = 0.01
        _tolerance = 3

        _p_cnt = int(os.environ['P_CNT'])

        if args.ada_mask:
            metric_past = float(os.environ['P_METRIC'])
            if metric_cur > metric_past + 0.02:
                if _p_cnt < _tolerance:
                    os.environ['P_CNT'] = str(_p_cnt + 1)
                else:
                    os.environ['MASKING_P'] = str(_p - _increment)
                    os.environ['P_CNT'] = str(0)
            elif metric_cur < metric_past - 0.02:
                if _p_cnt < _tolerance:
                    os.environ['P_CNT'] = str(_p_cnt + 1)
                else:
                    os.environ['MASKING_P'] = str(_p + _increment)
                    os.environ['P_CNT'] = str(0)
        elif args.entropy:
            entropy_past = float(os.environ['ENTROPY'])
            if entropy > entropy_past + 0.05:
                if _p_cnt < _tolerance:
                    os.environ['P_CNT'] = str(_p_cnt + 1)
                else:
                    os.environ['MASKING_P'] = str(_p - _increment)
                    os.environ['P_CNT'] = str(0)
            elif entropy < entropy_past - 0.05:
                if _p_cnt < _tolerance:
                    os.environ['P_CNT'] = str(_p_cnt + 1)
                else:
                    os.environ['MASKING_P'] = str(_p + _increment)
                    os.environ['P_CNT'] = str(0)
    os.environ['EVAL_CNT'] = str(int(os.environ['EVAL_CNT'])+1)
    return {'P_err': p_err, 'Token_acc': acc_token, 'Metric': metric_cur, 'RankMe': rankme, 'Entropy': entropy}


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CACHE_DIR'] = 'C:/Users/ay011/.cache/huggingface/datasets'
    os.environ['MASKING_P'] = str(args.p)

    os.environ['ITERATION_STEP'] = str(0)
    os.environ['EXP_NAME'] = '-'.join(
        ['lmcd', os.environ['WANDB_PROJECT'], str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))])

    os.environ['LOG_DIR'] = f'./logs/{os.environ["EXP_NAME"]}'
    os.mkdir(os.environ['LOG_DIR'])
    os.mkdir(os.path.join(os.environ['LOG_DIR'], 'batch'))

    np.random.seed(args.seed)

    _model_path = args.ckpt if args.ckpt is not None else args.model_type
    model = AutoModelForMaskedLM.from_pretrained(_model_path)

    train_args = {'learning_rate': args.lr, 'num_train_epochs': args.epochs, 'weight_decay': args.wd,
                  'per_device_train_batch_size': args.b_train, 'per_device_eval_batch_size': args.b_eval,
                  'do_eval': True, 'do_train': True,
                  'max_steps': args.max_steps,
                  'logging_steps': args.logging_steps, 'save_steps': args.save_steps}

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

    mlm_collator = CustomMLMCollator(tokenizer, args.p)

    print(dataset)
    acc, loss = evaluate(model, dataset)
    print(f'ACC {acc}')
    print(f'Loss {loss}')