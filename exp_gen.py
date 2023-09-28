import itertools

# !python main.py --model_type prajjwal1/bert-medium --p 0.2 --seed 89453 --max_steps 1000 --b_train 2 --b_eval 8 --data imdb --chunk_size 64 --data_type huggingface

# # seeds = [124, 78647, 45157, 35735, 89453, 56142, 99783, 13737, 24799, 59767]
# # seeds = [124, 78647, 45157, 35735, 89453]
# seeds = [45157, 35735, 89453]
# ps = [.2, .4, 0.6, 0.8]
# # ps = [.1, .3, 0.5, 0.7]
# # ps = [.1, .3, 0.5, 0.7, .2, .4, 0.6, 0.8]
#
# # dists = ['addition', 'nonlinear']
# # model_types = ['prajjwal1/bert-tiny', 'prajjwal1/bert-mini', 'prajjwal1/bert-medium', 'bert-base-cased', 'bert-large-uncased']
# # model_types = ['prajjwal1/bert-tiny', 'prajjwal1/bert-mini', 'prajjwal1/bert-medium', 'bert-base-cased']
# model_types = ['bert-base-cased']
# # model_types = ['prajjwal1/bert-medium']
#
# default_op = '--max_steps 20000 --b_train 8 --b_eval 8 --shard_eval 300 --logging_steps 100 --data bookcorpus --chunk_size 64 --data_type huggingface\n'
# with open('exp_base.bat', 'w') as f:
#     for p, model_type, seed in itertools.product([ 0.6, 0.8], model_types, [78647]):
#         f.write(
#             f'python main.py --model_type {model_type} --p {p} --seed {seed} {default_op}')
#     for p, model_type, seed in itertools.product(ps, model_types, seeds):
#         f.write(f'python main.py --model_type {model_type} --p {p} --seed {seed} {default_op}')
#

with open('squad_script.bat', 'w') as f:
    # steps = list(range(1, 17))
    # steps = steps[::-1]
    # steps = list(range(1, 20))[::-1]
    # models = ['const', 'step', 'cosine', 'adaMemo', 'adaToken']

    steps = list(range(1, 7))[::-1]
    models = ['adaTokenadaToken']

    for i, model_type in itertools.product(steps, models):
        f.write(f'python run_squad.py --ckpt ./ckpts/bert-tiny-p20-{model_type}-v8/checkpoint-{i*8}0000 --logging_steps 20001\n')
        # print(f'python run_squad.py --ckpt ./ckpts/bert-tiny-p20-{model_type}-v8/checkpoint-{i*4}000 --logging_steps 20001')
