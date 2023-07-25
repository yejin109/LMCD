import torch
import numpy as np
import matplotlib.pyplot as plt


for n in range(2, 11):
    dataset_size = 10
    seq_len = n
    # seq_len = 5

    # vocab_size = v
    vocab_size = 2
    tolerance = 0.05
    # tolerance = 1/vocab_size

    # p_y = 1/(vocab_size)
    # h_y = -np.log(p_y)

    # masked_tokens = np.arange(1, seq_len)
    # mask_p = masked_tokens / seq_len

    mask_p = np.arange(0.01, 0.99, step=0.001)
    masked_tokens = seq_len*mask_p

    h_xn = seq_len * np.log2(vocab_size)

    # h_xnyn = vocab_size**(seq_len*(1-mask_p))*seq_len*mask_p*np.log2(vocab_size)
    h_xnyn = np.ceil(seq_len*mask_p)*np.log2(vocab_size)
    # h_xnyn = seq_len*mask_p*np.log2(vocab_size)

    # p_yx = mask_p**masked_tokens * (1-mask_p)**(seq_len - masked_tokens)
    # h_yx = p_yx*np.log2(1/p_yx) + (1-p_yx)*np.log2(1/(1-p_yx))

    # mi = h_y - h_yx

    mi = h_xn - h_xnyn

    # thr = 2**(-seq_len * mi)
    # thr = tolerance/(seq_len*dataset_size) * 2**(seq_len*mi)

    upper_bound = np.ceil(seq_len*mask_p)*dataset_size*2**(-seq_len*mask_p*mi)
    # upper_bound = seq_len*mask_p*dataset_size*2**(-seq_len*mask_p*mi)

    # p_err = 2**(np.log(seq_len*mask_p)-seq_len*mi)
    # p_err = seq_len*mask_p*dataset_size*2**(-seq_len*mi)

    # dp_err_d_p = (upper_bound[1:]-upper_bound[:-1])/(mask_p[1:]-mask_p[:-1])

    # p_err[p_err>=1] = 1.
    # p_threshold_idx = np.argwhere(upper_bound < tolerance)

    plt.figure()
    # plt.plot(mask_p, p_err, label='P_err')
    plt.plot(mask_p, upper_bound, label='P err Upper bound')
    plt.plot(mask_p, np.ones_like(mask_p)*tolerance, label='Tolerance')
    plt.title(f"seq_len : {seq_len} vocab size : {vocab_size}")
    # plt.title(f"seq_len : {seq_len} vocab size : {vocab_size}")
    plt.legend()
    plt.ylim(0, 1+.1)
    plt.savefig(f'./results/v_{vocab_size}_n_{seq_len}.png')
    # plt.savefig(f'./results/v_variation_{str(vocab_size).zfill(2)}.png')
    # plt.show()
    plt.close()
