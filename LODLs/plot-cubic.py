#!/usr/bin/env python3

import argparse

import os
import sys
import pickle as pkl
import matplotlib.pyplot as plt

from IPython.core import ultratb
sys.excepthook = ultratb.FormattedTB(mode='Verbose',
                                        color_scheme='Linux',
                                        call_pdb=1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_root', type=str)
    parser.add_argument('--pkl_tag', type=str, default='latest')
    args = parser.parse_args()

    exp_path = f'{args.exp_root}/{args.pkl_tag}.pkl'
    assert os.path.exists(exp_path)
    print('-- loading exp')
    with open(exp_path, 'rb') as f:
        exp = pkl.load(f)
    print('-- done')

    # import ipdb; ipdb.set_trace()
    X_test, Y_test, Y_test_aux = exp.problem.get_test_data()
    fig, ax = plt.subplots()
    ax.scatter(X_test[0].ravel(), Y_test[0].ravel())
    import ipdb; ipdb.set_trace()
    fig.savefig('t.png')

if __name__ == '__main__':
    main()
