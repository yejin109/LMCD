from datasets import load_dataset
from datasets import Dataset
from .synthetic import SynthDataSet
from .collator import custom_mlm_collator


def get_dataset(_data_type, data=None):
    if _data_type == 'synthetic':
        return Dataset.from_dict(data)
    else:
        return load_dataset(data)


def get_data(_data_args):
    data = SynthDataSet(*_data_args)
    return data

