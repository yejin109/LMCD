import datetime
import os
import argparse

from data import get_dataset, get_data, CustomMLMCollator
from _utils import CustomWandbCallback
import numpy as np
from transformers import AutoModelForMaskedLM, TrainingArguments, Trainer, AutoTokenizer
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default="bert-base-cased")
parser.add_argument('--data_type', default='huggingface')
parser.add_argument('--data', default='imdb', required=False)

# Synthetic data
parser.add_argument('--v', default=2, type=int)
parser.add_argument('--dist', default='SRS',
                    help="How to inject non iid",
                    choices=['uniform', 'SRS', 'addition', 'nonlinear'])
parser.add_argument('--seed', default=123, type=int)
parser.add_argument('--n', default=10, type=int)
parser.add_argument('--D', default=10000)
parser.add_argument('--p', default=.20, type=float)
parser.add_argument('--chunk_size', default=64, type=int)

# train
parser.add_argument('--lr', default=2e-5)
parser.add_argument('--epochs', default=30)
parser.add_argument('--wd', default=1e-2)
parser.add_argument('--max_steps', type=int, default=1000)
parser.add_argument('--b_train', default=2, type=int)

# Test
parser.add_argument('--b_eval', default=2, type=int)
parser.add_argument('--shard_eval', default=600, type=int)

# Log
parser.add_argument('--logging_steps', default=10)
parser.add_argument('--save_steps', default=1000)


def train(_model, _dataset, _train_args, _tk, sharding_size=600):
    training_args = TrainingArguments(
        output_dir=os.environ['LOG_DIR'],
        evaluation_strategy="steps",  # candidates : steps, epochs
        run_name=os.environ['EXP_NAME'] + '-'.join(
            ['', os.environ['MASKING_P'], str(args.seed), str(args.n), str(args.v)]),
        **_train_args,
    )

    mlm_collator = CustomMLMCollator(_tk, args.p)

    trainer = Trainer(
        model=_model,
        args=training_args,
        train_dataset=_dataset["train"],
        eval_dataset=_dataset["test"].shard(num_shards=sharding_size, index=np.random.randint(0, sharding_size, size=1)),
        data_collator=mlm_collator,
        compute_metrics=compute_metrics,

    )
    trainer.add_callback(CustomWandbCallback)

    trainer.train()


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
    # os.environ['WANDB_PROJECT'] = args.dist + ' - theory'
    os.environ['WANDB_PROJECT'] = args.data

    os.environ['ITERATION_STEP'] = str(0)
    os.environ['EXP_NAME'] = '-'.join(
        ['lmcd', os.environ['WANDB_PROJECT'], str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))])

    os.environ['LOG_DIR'] = f'./logs/{os.environ["EXP_NAME"]}'
    os.mkdir(os.environ['LOG_DIR'])
    os.mkdir(os.path.join(os.environ['LOG_DIR'], 'batch'))

    np.random.seed(args.seed)

    model = AutoModelForMaskedLM.from_pretrained(args.model_type)

    train_args = {'learning_rate': args.lr, 'num_train_epochs': args.epochs, 'weight_decay': args.wd,
                  'per_device_train_batch_size': args.b_train, 'per_device_eval_batch_size': args.b_eval,
                  'do_eval': True, 'do_train': True,
                  'max_steps': args.max_steps,
                  'logging_steps': args.logging_steps, 'save_steps': args.save_steps}

    if args.data_type == 'synthetic':
        data = get_data((args.v, args.dist, args.n, args.D, args.b, args.seed))
        dataset = get_dataset(args.data_type, data.get_data())
        dataset.set_format(type='torch', columns=['input_ids'])
        dataset = dataset.train_test_split(args.test_ratio)
        tokenizer = None
    elif args.data_type == 'huggingface':
        dataset = get_dataset(args.data_type, args.data)
        if 'unsupervised' in dataset.keys():
            del dataset['unsupervised']
        tokenizer = AutoTokenizer.from_pretrained(args.model_type)
        tokenized_datasets = dataset.map(tokenize_function,
                                         batched=True, remove_columns=list(dataset.features.keys()),
                                         fn_kwargs={'_tokenizer': tokenizer})
        chunk_size = args.chunk_size
        lm_datasets = tokenized_datasets.map(group_texts,
                                             batched=True,
                                             fn_kwargs={'_chunk_size': chunk_size})
        lm_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        dataset = lm_datasets
        tokenized_datasets = None

        # Masking test
        # dataset = lm_datasets['train']
        # samples = [dataset[i] for i in range(2)]
        # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
        #
        # for chunk in custom_mlm_collator(samples)["input_ids"]:
        # for chunk in data_collator(samples)["input_ids"]:
        #     print(f"\n'>>> {tokenizer.decode(chunk)}'")
    else:
        dataset = None
        raise AssertionError(f"")

    print(dataset)

    train(model, dataset, train_args, tokenizer, sharding_size=args.shard_eval)
