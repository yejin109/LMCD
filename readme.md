# setup

- hugging face 
    ```
    pip install transformers[torch]
    pip install datasets
    ```
- cuda: CUDA 11.8


# Synthetic data

Because synthetic data is composed of a number of integers, current version does not require tokenizer. As a result, dataset interface for synthetic data generate whole dataset with size of (size, seq len)

Once we populate systhetic dataset, we use dataset interface from huggingface so that we can use common workflow from the huggingface framework. 

## Special tokens information
  - [MASK] : 0 
