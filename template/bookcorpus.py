import os
from datasets import load_dataset, DatasetDict
from transformers import PretrainedConfig, AutoConfig


def get_dataset(data, train_test_split=0.0005, only_test=False):
    kwargs = {}
    dataset = load_dataset(data, cache_dir=os.environ['CACHE_DIR'], **kwargs)
    if isinstance(dataset, dict) and ('test' not in dataset.keys()):
        dataset = dataset['train'].train_test_split(train_test_split)
    if only_test:
        dataset = dataset['test']
        dataset = DatasetDict({'test': dataset})
    return dataset


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


def preprocess(dataset, tokenizer):
    tokenized_datasets = dataset.map(tokenize_function,
                                     batched=True, remove_columns=list(dataset['test'].features.keys()),
                                     fn_kwargs={'_tokenizer': tokenizer})
    chunk_size = min(512, tokenizer.model_max_length)
    lm_datasets = tokenized_datasets.map(group_texts,
                                         batched=True,
                                         fn_kwargs={'_chunk_size': chunk_size})
    lm_datasets = lm_datasets
    lm_datasets.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    return lm_datasets
