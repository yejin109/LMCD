import matplotlib.pyplot as plt
import pickle
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='imdb')
parser.add_argument('--p', type=float, default=.2)
parser.add_argument('--chunk_size', type=int, default=128)


if __name__ == '__main__':
    args = parser.parse_args()
    plt.figure()
    with open(f'./results/{args.data}_chunk{args.chunk_size}_p{int(args.p*100)}.pkl') as f:
