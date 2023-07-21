import _init_env
import datetime
import os
import argparse
from data import get_dataset, get_data, custom_mlm_collator
from _utils import CustomWandbCallback

import numpy as np
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default="bert-base-cased")
parser.add_argument('--data_type', default='synthetic')
parser.add_argument('--dataset', required=False)

# Synthetic data
parser.add_argument('--v', default=6)
parser.add_argument('--dist', default='uniform',
                    help="How to inject non iid",
                    choices=['uniform', 'SRS', 'addition', 'nonlinear'])
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--n', default=5)
parser.add_argument('--D', default=10000)
parser.add_argument('--test_ratio', default=0.1)
parser.add_argument('--b', default=512)
parser.add_argument('--p', default=.2, type=float)

# train
parser.add_argument('--lr', default=2e-5)
parser.add_argument('--epochs', default=30)
parser.add_argument('--wd', default=1e-2)

# Log
parser.add_argument('--logging_steps', default=10)
parser.add_argument('--save_steps', default=1000)


def train(_model, _dataset, _train_args):
    training_args = TrainingArguments(
        output_dir=os.environ['LOG_DIR'],
        evaluation_strategy="steps", # candidates : steps, epochs
        run_name=os.environ['EXP_NAME']+'-'.join(['', os.environ['MASKING_P'], str(args.seed)]),
        **_train_args,
    )

    trainer = Trainer(
        model=_model,
        args=training_args,
        train_dataset=_dataset["train"],
        eval_dataset=_dataset["test"],
        data_collator=custom_mlm_collator,
        compute_metrics=compute_metrics
    )
    trainer.add_callback(CustomWandbCallback)

    trainer.train()


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    preds = np.argmax(preds, -1)
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)
    mask = labels != -100

    labels = labels[mask]
    preds = preds[mask]

    return {'P_err': 1 - accuracy_score(labels, preds)}


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['MASKING_P'] = str(args.p)
    os.environ['LOGGING_STEP'] = str(args.logging_steps)
    os.environ['VOCAB_SIZE'] = str(args.v)
    os.environ['SEQ_LEN'] = str(args.n)
    os.environ['WANDB_PROJECT'] = args.dist + ' - expand'

    data = get_data((args.v, args.dist, args.n, args.D, args.b, args.seed))
    dataset = get_dataset(args.data_type, data.get_data())
    dataset.set_format(type='torch', columns=['input_ids'])
    dataset = dataset.train_test_split(args.test_ratio)

    model = AutoModelForMaskedLM.from_pretrained(args.model_type)

    train_args = {'learning_rate': args.lr, 'num_train_epochs': args.epochs, 'weight_decay': args.wd,
                  'per_device_train_batch_size': args.b, 'per_device_eval_batch_size': args.b,
                  'do_eval': True, 'do_train': True,
                  'logging_steps': args.logging_steps, 'save_steps': args.save_steps}
    train(model, dataset, train_args)
