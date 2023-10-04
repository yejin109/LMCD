python run_squad.py --ckpt ./ckpts/bert-tiny-p20-step-v9/checkpoint-200000 --logging_steps 1000 --seed 78647 --epochs 6
python run_squad.py --ckpt ./ckpts/bert-tiny-p20-step-v9/checkpoint-200000 --logging_steps 1000 --seed 45157 --epochs 6
python run_squad.py --ckpt ./ckpts/bert-tiny-p20-step-v9/checkpoint-200000 --logging_steps 1000 --seed 35735 --epochs 6
python run_squad.py --ckpt ./ckpts/bert-tiny-p20-step-v9/checkpoint-200000 --logging_steps 1000 --seed 89453 --epochs 6
python run_squad.py --ckpt ./ckpts/bert-tiny-p20-cosine-v9/checkpoint-200000 --logging_steps 1000 --seed 78647 --epochs 6
python run_squad.py --ckpt ./ckpts/bert-tiny-p20-cosine-v9/checkpoint-200000 --logging_steps 1000 --seed 45157 --epochs 6
python run_squad.py --ckpt ./ckpts/bert-tiny-p20-cosine-v9/checkpoint-200000 --logging_steps 1000 --seed 35735 --epochs 6
python run_squad.py --ckpt ./ckpts/bert-tiny-p20-cosine-v9/checkpoint-200000 --logging_steps 1000 --seed 89453 --epochs 6
python run_squad.py --ckpt ./ckpts/bert-tiny-p20-adaMemo-v9/checkpoint-200000 --logging_steps 1000 --seed 78647 --epochs 6
python run_squad.py --ckpt ./ckpts/bert-tiny-p20-adaMemo-v9/checkpoint-200000 --logging_steps 1000 --seed 45157 --epochs 6
python run_squad.py --ckpt ./ckpts/bert-tiny-p20-adaMemo-v9/checkpoint-200000 --logging_steps 1000 --seed 35735 --epochs 6
python run_squad.py --ckpt ./ckpts/bert-tiny-p20-adaMemo-v9/checkpoint-200000 --logging_steps 1000 --seed 89453 --epochs 6

python run_squad.py --ckpt ./ckpts/bert-tiny-p20-const-v9/checkpoint-200000 --logging_steps 1000 --seed 35735 --epochs 6
python run_squad.py --ckpt ./ckpts/bert-tiny-p20-const-v9/checkpoint-200000 --logging_steps 1000 --seed 89453 --epochs 6

python main.py --data bookcorpus --chunk_size 64 --b_train 128 --epochs 2
REM python main.py --ada_token --data bookcorpus --chunk_size 64 --b_train 128 --epochs 2
REM python main.py --cosine --data bookcorpus --chunk_size 64 --b_train 128 --epochs 2
REM python main.py --ada_memo --data bookcorpus --chunk_size 64 --b_train 128 --epochs 2
REM python main.py --step --data bookcorpus --chunk_size 64 --b_train 128 --epochs 2

REM python main.py --ckpt ./ckpts/bert-tiny-p20-const-v8/checkpoint-136000
REM python main.py --step --ckpt ./ckpts/bert-tiny-p20-step-v8/checkpoint-136000 --logging_steps 2000 --save_steps 20000
REM python main.py --cosine --ckpt ./ckpts/bert-tiny-p20-cosine-v8/checkpoint-136000 --logging_steps 2000 --save_steps 20000
REM python main.py --ada_token --ckpt ./ckpts/bert-tiny-p20-adaToken-v8/checkpoint-136000 --logging_steps 2000 --save_steps 20000
REM python main.py --ada_memo --ckpt ./ckpts/bert-tiny-p20-adaMemo-v8/checkpoint-136000 --logging_steps 2000 --save_steps 20000
REM python main.py --p 0.4
REM python main.py --mrd

REM python main.py --data wikipedia --chunk_size 128 --b_train 64 --ckpt ./ckpts/bert-tiny-p20-const-v9/checkpoint-260000
REM python main.py --ada_token --data wikipedia --chunk_size 128 --b_train 64 --ckpt ./ckpts/bert-tiny-p20-adaToken-v9/checkpoint-260000


