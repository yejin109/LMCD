# setup

- hugging face 
    ```
    pip install transformers[torch]
    pip install datasets
    pip install chardet
    ```
  
  현재 Data Collator 사용을 위해서 pip install로 지원하지 않은 class 사용하기 위해서 당므 코드로 설치하기로 한다.
  ```
  git clone https://github.com/huggingface/transformers
  ```
- cuda: CUDA 11.8



# Synthetic data

현재 iid하게 사용할 경우 tokenizer를 사용하지 않는 방식으로 구현.
그렇기에 각 token별 id를 실제 데이터와 맞춰서 구현할 때 유의해야 한다.

## Special tokens
  - [MASK] : 0 