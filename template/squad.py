import os
import math

from transformers import TrainingArguments, DefaultDataCollator, EvalPrediction
from qa import QuestionAnsweringTrainer, prepare_validation_features, postprocess_qa_predictions
from datasets import load_metric
from _utils import CustomWandbCallback

DATA = 'squad'
_metric = load_metric(DATA)
question_column_name = None
context_column_name = None
answer_column_name = None


def preprocess_function(examples, tokenizer):
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


def set_column_name(cols):
    global question_column_name, context_column_name, answer_column_name

    question_column_name = "question" if "question" in cols else cols[0]
    context_column_name = "context" if "context" in cols else cols[1]
    answer_column_name = "answers" if "answers" in cols else cols[2]


def compute_metric(p: EvalPrediction):
    return _metric.compute(predictions=p.predictions, references=p.label_ids)


def pipeline_squad_data(dataset, tokenizer, chunk_size):
    column_names = dataset["test"].column_names
    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

    eval_kwargs = {
        'max_seq_length': chunk_size,
        'question_column_name': question_column_name,
        'context_column_name': context_column_name,
        'tokenizer': tokenizer
    }
    eval_dataset = dataset['test'].map(
        prepare_validation_features,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on validation dataset",
        fn_kwargs=eval_kwargs
    )
    eval_dataset = eval_dataset.shard(int(math.floor(len(eval_dataset) / 5000)), index=0)
    tokenized_datasets['test'] = eval_dataset

    return tokenized_datasets


def pipeline_squad_train(_model, _dataset, _train_args, _tk, eval_examples):
    training_args = TrainingArguments(
        output_dir=os.environ['LOG_DIR'],
        evaluation_strategy="steps",  # candidates : steps, epoch
        run_name=os.environ['RUN_NAME'],
        **_train_args,
    )

    data_collator = DefaultDataCollator()

    trainer = QuestionAnsweringTrainer(
        model=_model,
        args=training_args,
        train_dataset=_dataset["train"],
        eval_dataset=_dataset["test"],
        eval_examples=eval_examples,
        tokenizer=_tk,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metric
    )
    trainer.add_callback(CustomWandbCallback)

    trainer.train()


def pipeline_squad(_dataset, _tokenizer, _chunk_size, _model, _train_args):
    set_column_name(_dataset["test"].column_names)
    _dataset = pipeline_squad_data(_dataset, _tokenizer, _chunk_size)
    pipeline_squad_train(_model, _dataset, _train_args, _tokenizer, eval_examples=_dataset['test'])
