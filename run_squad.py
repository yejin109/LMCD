import datetime
import os
import argparse
import math
from data import get_dataset
from _utils import CustomWandbCallback
import numpy as np
from transformers import TrainingArguments, AutoTokenizer, DefaultDataCollator, EvalPrediction, AutoModelForQuestionAnswering
from datasets import load_metric
from qa import postprocess_qa_predictions, QuestionAnsweringTrainer, prepare_validation_features


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=123, type=int)

parser.add_argument('--model_type', default="prajjwal1/bert-medium")
parser.add_argument('--ckpt', default="./logs/bert-medium-p20-ada-v5/checkpoint-20000")

# parser.add_argument('--model_type', default="bert-base-cased")
# parser.add_argument('--ckpt', default="./logs/bert-base-p40-const/checkpoint-20000")

parser.add_argument('--data_type', default='huggingface')
parser.add_argument('--split_load', default=10000, type=int)
parser.add_argument('--data', default='squad', required=False, help='default squad')
parser.add_argument('--split_test', default=0.2, type=float)

parser.add_argument('--chunk_size', default=64, type=int)

# train
parser.add_argument('--lr', default=2e-5)
parser.add_argument('--epochs', default=3, help='num_train_epochs')
parser.add_argument('--wd', default=1e-2, help='weight decay')
parser.add_argument('--max_steps', type=int, default=20000)
parser.add_argument('--b_train', default=8, type=int)

# Test
parser.add_argument('--b_eval', default=8, type=int)
parser.add_argument('--shard_eval', default=300, type=int)

# Log
parser.add_argument('--logging_steps', type=int, default=50)
parser.add_argument('--save_steps', type=int, default=20000)


def train(_model, _dataset, _train_args, _tk, eval_examples):
    _run_name = [args.ckpt.split('/')[2], str(args.seed)]

    training_args = TrainingArguments(
        output_dir=os.environ['LOG_DIR'],
        evaluation_strategy="steps",  # candidates : steps, epoch
        run_name='-'.join(_run_name),
        **_train_args,
    )

    data_collator = DefaultDataCollator()

    trainer = QuestionAnsweringTrainer(
        model=_model,
        args=training_args,
        train_dataset=_dataset["train"],
        eval_dataset=_dataset["test"],
        eval_examples=eval_examples,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metric
    )
    trainer.add_callback(CustomWandbCallback)

    trainer.train()


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def post_processing_function(examples, features, predictions, stage="eval"):
    # Post-processing: we match the start logits and end logits to answers in the original context.
    predictions = postprocess_qa_predictions(
        examples=examples,
        features=features,
        predictions=predictions,
        # version_2_with_negative=data_args.version_2_with_negative,
        # n_best_size=data_args.n_best_size,
        # max_answer_length=data_args.max_answer_length,
        # null_score_diff_threshold=data_args.null_score_diff_threshold,
        # output_dir=training_args.output_dir,
        # log_level=log_level,
        prefix=stage,
    )
    # Format the result to the format the metric expects.
    formatted_predictions = [{"id": str(k), "prediction_text": v} for k, v in predictions.items()]

    references = [{"id": str(ex["id"]), "answers": ex[answer_column_name]} for ex in examples]
    return EvalPrediction(predictions=formatted_predictions, label_ids=references)


def compute_metric(p: EvalPrediction):
    return metric.compute(predictions=p.predictions, references=p.label_ids)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CACHE_DIR'] = 'C:/Users/ay011/.cache/huggingface/datasets'
    os.environ['LOGGING_STEP'] = str(args.logging_steps)
    os.environ['WANDB_PROJECT'] = args.data + ' - v5'

    os.environ['ITERATION_STEP'] = str(0)
    os.environ['EXP_NAME'] = '-'.join(
        ['lmcd', os.environ['WANDB_PROJECT'], str(datetime.datetime.now().strftime("%y%m%d-%H%M%S"))])

    os.environ['LOG_DIR'] = f'./logs/{os.environ["EXP_NAME"]}'
    os.mkdir(os.environ['LOG_DIR'])
    os.mkdir(os.path.join(os.environ['LOG_DIR'], 'batch'))

    np.random.seed(args.seed)
    model = AutoModelForQuestionAnswering.from_pretrained(args.ckpt)
    tokenizer = AutoTokenizer.from_pretrained(args.model_type)

    train_args = {'learning_rate': args.lr, 'num_train_epochs': args.epochs, 'weight_decay': args.wd,
                  'per_device_train_batch_size': args.b_train, 'per_device_eval_batch_size': args.b_eval,
                  'do_eval': True, 'do_train': True,
                  'logging_steps': args.logging_steps, 'save_steps': args.save_steps}

    dataset = get_dataset(args.data_type, args.data, split=args.split_load)
    dataset = dataset.train_test_split(args.split_test)
    column_names = dataset["test"].column_names
    answer_column_name = "answers" if "answers" in column_names else column_names[2]
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)
    eval_dataset = dataset['test'].map(
                prepare_validation_features,
                batched=True,
                remove_columns=column_names,
                desc="Running tokenizer on validation dataset",
                fn_kwargs={'max_seq_length': args.chunk_size, 'question_column_name': question_column_name, 'context_column_name':context_column_name, 'tokenizer': tokenizer}
            )
    eval_dataset = eval_dataset.shard(int(math.floor(len(eval_dataset)/5000)), index=0)
    tokenized_datasets['test'] = eval_dataset
    print(tokenized_datasets)
    metric = load_metric(args.data)
    train(model, tokenized_datasets, train_args, tokenizer, eval_examples=dataset['test'])
