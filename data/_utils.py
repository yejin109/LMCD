from huggingface_hub import list_datasets


def get_dataset_list(_max_cnt=10):
    _cnt = 0
    # Return DatasetInfo
    dataset_list_generator = list_datasets()
    while True:
        if _cnt >= _max_cnt:
            break
        print(next(dataset_list_generator))
        _cnt += 1
