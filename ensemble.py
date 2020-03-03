import argparse
from collections import Counter

import numpy as np

from utils import save_result, load_result


def vote_once(pred, default):
    most_common = Counter(pred).most_common(1)
    if most_common[0][1] > len(pred) // 2:
        return most_common[0][0]
    return default


def vote(pred_mat, baseline):
    n_samples = pred_mat.shape[0]
    result = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        result[i] = vote_once(pred_mat[i], baseline[i])
    return result


def ensamble(inputs, baseline_result):
    pred = np.zeros((200000, len(inputs)), dtype=int)
    for i, result in enumerate(inputs):
        pred[:, i] = load_result(result)
    baseline = load_result(baseline_result)
    return vote(pred, baseline)


def parse_args():
    parser = argparse.ArgumentParser(description='Ensemble results')
    parser.add_argument('inputs', nargs='+',
                        help='inputs to ensemble')
    parser.add_argument('-b', '--baseline', required=True,
                        help='the baseline used to resolve conflict')
    parser.add_argument('-o', '--output', default='ensemble_result.csv',
                        help='result of ensemble')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = ensamble(args.inputs, args.baseline)
    save_result(args.output, result)
