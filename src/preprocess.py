from transformers import AutoModel, AutoTokenizer


def get_tokenizer(pretrained_model_name_or_path):
    return AutoTokenizer.from_pretrained(pretrained_model_name_or_path=pretrained_model_name_or_path)


if __name__ == '__main__':
    model_type = "bert-base-cased"
    tokenizer = get_tokenizer(model_type)
    print()
