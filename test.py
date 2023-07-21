import itertools

seeds = [124, 78647, 45157, 35735, 89453, 56142, 99783, 13737, 24799, 59767]
ps = [ 0.15, 0.5, 0.75, 0.9, 0.99]
dists = ['addition', 'nonlinear']
with open('exp.bat', 'w') as f:
    for dist, seed, p in itertools.product(dists, seeds, ps):
        f.write(f'python main.py --dist {dist} --seed {seed} --p {p}\n')
