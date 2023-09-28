REM python run_squad.py --ckpt ./ckpts/bert-tiny-p20-adaTokenadaToken-v8/checkpoint-460000 --logging_steps 20001
REM python run_squad.py --ckpt ./ckpts/bert-tiny-p20-adaTokenadaToken-v8/checkpoint-400000 --logging_steps 20001
REM python run_squad.py --ckpt ./ckpts/bert-tiny-p20-adaTokenadaToken-v8/checkpoint-320000 --logging_steps 20001
REM python run_squad.py --ckpt ./ckpts/bert-tiny-p20-adaTokenadaToken-v8/checkpoint-240000 --logging_steps 20001
REM python run_squad.py --ckpt ./ckpts/bert-tiny-p20-adaTokenadaToken-v8/checkpoint-160000 --logging_steps 20001
REM python run_squad.py --ckpt ./ckpts/bert-tiny-p20-adaTokenadaToken-v8/checkpoint-80000 --logging_steps 20001

REM python main.py --ckpt ./ckpts/bert-tiny-p20-const-v8/checkpoint-136000
REM python main.py --step --ckpt ./ckpts/bert-tiny-p20-step-v8/checkpoint-136000 --logging_steps 2000 --save_steps 20000
REM python main.py --cosine --ckpt ./ckpts/bert-tiny-p20-cosine-v8/checkpoint-136000 --logging_steps 2000 --save_steps 20000
REM python main.py --ada_token --ckpt ./ckpts/bert-tiny-p20-adaToken-v8/checkpoint-136000 --logging_steps 2000 --save_steps 20000
REM python main.py --ada_memo --ckpt ./ckpts/bert-tiny-p20-adaMemo-v8/checkpoint-136000 --logging_steps 2000 --save_steps 20000
REM python main.py --p 0.4
REM python main.py --mrd

python main.py --data bookcorpus --chunk_size 64 --b_train 128 --epochs 2
python main.py --ada_token --data bookcorpus --chunk_size 64 --b_train 128 --epochs 2
python main.py --ada_memo --data bookcorpus --chunk_size 64 --b_train 128 --epochs 2
REM python main.py --cosine --data bookcorpus --chunk_size 64 --b_train 128 --epochs 4