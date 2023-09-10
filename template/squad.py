import math
from qa import prepare_validation_features


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


def pipeline_squad(dataset, tokenizer, chunk_size):
    column_names = dataset["test"].column_names
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    # answer_column_name = "answers" if "answers" in column_names else column_names[2]

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
    eval_dataset = eval_dataset.shard(int(math.floor(len(eval_dataset)/5000)), index=0)
    tokenized_datasets['test'] = eval_dataset