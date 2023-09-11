from transformers import AutoModelForMaskedLM, AutoTokenizer


def get_tokenizer(_model_type):
    return AutoTokenizer.from_pretrained(_model_type)


def get_model(_model_type):
    return AutoModelForMaskedLM.from_pretrained(_model_type)
