import os
from datasets import load_dataset
from datasets import Dataset
from .synthetic import SynthDataSet
from .collator import custom_mlm_collator, CustomMLMCollator


def get_dataset(_data_type, data=None):
    if _data_type == 'synthetic':
        return Dataset.from_dict(data)
    elif _data_type == 'huggingface':
        if data == 'wikipedia':
            res = load_dataset(data, '20220301.en', cache_dir=os.environ['CACHE_DIR'])
            res = res.remove_columns(["id", 'title', 'url'])
            return res
        else:
            return load_dataset(data, cache_dir=os.environ['CACHE_DIR'])

    else:
        raise AssertionError(f"Cannot support {_data_type}")


def get_data(_data_args):
    data = SynthDataSet(*_data_args)
    return data

