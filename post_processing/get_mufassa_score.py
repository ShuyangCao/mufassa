import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original')
    parser.add_argument('--empty')
    parser.add_argument('--mask')
    parser.add_argument('--shuffle')
    parser.add_argument('--pos')
    parser.add_argument('--start')
    args = parser.parse_args()

    with open(args.original) as f:
        original_probs = [[float(prob) for prob in line.strip().split()] for line in f]

    with open(args.empty) as f:
        empty_probs = [[float(prob) for prob in line.strip().split()] for line in f]

    with open(args.mask) as f:
        mask_probs = [[float(prob) for prob in line.strip().split()] for line in f]

    with open(args.shuffle) as f:
        shuffle_probs = [[float(prob) for prob in line.strip().split()] for line in f]

    with open(args.pos) as f:
        poss = [line.strip().split() for line in f]

    with open(args.start) as f:
        starts = [line.strip().split() for line in f]

    unimportant_pos_tags = ['PUNCT', 'SYM', 'DET', 'PART', 'CCONJ', 'SCONJ']
    original_empty_diff = [[op - ep for op, ep in zip(original_prob, empty_prob)] for original_prob, empty_prob in zip(original_probs, empty_probs)]
    original_mask_diff = [[op - ep for op, ep in zip(original_prob, empty_prob)] for original_prob, empty_prob in
                          zip(original_probs, mask_probs)]
    original_shuffle_diff = [[op - ep for op, ep in zip(original_prob, empty_prob)] for original_prob, empty_prob in
                             zip(original_probs, shuffle_probs)]
    original_prob_scores = np.array([
        np.mean(
            [md if p in ['PROPN', 'NUM'] else sd if p in ['VERB', 'ADJ', 'ADV', 'ADP'] else ed for ed, md, sd, p, s in
             zip(empty_diffs, mask_diffs, shuffle_diffs, pos, start) if p not in unimportant_pos_tags and s == '1']) for
        empty_diffs, mask_diffs, shuffle_diffs, pos, start in
        zip(original_empty_diff, original_mask_diff, original_shuffle_diff, poss, starts)])

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.output_dir, 'mufassa_single.txt'), 'w') as f:
        for prob in original_prob_scores:
            f.write(str(prob) + '\n')

    with open(os.path.join(args.output_dir, 'mufassa.txt'), 'w') as f:
        f.write(str(np.mean(original_prob_scores)) + '\n')


if __name__ == '__main__':
    main()
