from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AutoConfig


def get_tokenizer(_model_type):
    return DistilBertTokenizer.from_pretrained(_model_type)


def get_model(_model_type):
    return DistilBertForSequenceClassification.from_pretrained(_model_type)
