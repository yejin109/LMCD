import torch
import numpy.random as random
from collections import defaultdict


class SynthDataGenerator:
    def __init__(self, _vocab_size, _dist, _seed):
        self.vocab_size = _vocab_size

        self.seed = None
        self.set_seed(_seed)
        if _dist == 'uniform':
            self.generator = random.randint
        else:
            assert False, f"Cannot support {_dist}"

    def gen(self, size):
        return self.generator(low=1, high=self.vocab_size+1, size=size)

    def set_seed(self, seed):
        random.seed(seed)
        self.seed = seed


class SynthDataSet:
    def __init__(self, _vocab_size, _dist, _seq_len, _size, _batch_size, _seed=123):
        self.generator = SynthDataGenerator(_vocab_size, _dist, _seed)
        self.seq_len = _seq_len
        self.size = _size
        self.data_dict = defaultdict(list)
        self.gen()

    def gen(self):
        self.data_dict['input_ids'] = torch.Tensor(list(map(lambda x: self.generator.gen(self.seq_len), range(self.size)))).long()

    def get_data(self):
        return self.data_dict

    def __getitem__(self, item):
        return self.data_dict[item]


if __name__ == '__main__':
    vocabulary_size = 10
    distribution = 'uniform'
    seed = 123
    n = 10
    D_train = 10000
    D_val = 1000
    D_test = 1000
    dataset = SynthDataSet(vocabulary_size, distribution, n, D_train, D_val, D_test)
