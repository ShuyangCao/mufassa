import argparse
import os
import math
from fairseq.data.encoders.gpt2_bpe_utils import get_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate_dir', nargs='+')
    args = parser.parse_args()

    bpe_encoder = get_encoder(
        os.path.join(os.getenv('EXPDIR'), 'pretrain_language_models', 'fairseq.gpt2', 'encoder.json'),
        os.path.join(os.getenv('EXPDIR'), 'pretrain_language_models', 'fairseq.gpt2', 'vocab.bpe')
    )

    for data_split in ['train', 'valid', 'test']:
        for generate_dir in args.generate_dir:
            all_samples = []
            sample = []

            if not os.path.exists(os.path.join(generate_dir, f'generate-{data_split}.txt')):
                continue
            with open(os.path.join(generate_dir, f'generate-{data_split}.txt')) as f:
                for line in f:
                    if line[0] == 'S':
                        if sample:
                            all_samples.append((sample_id, sample))
                            sample = []
                        try:
                            sample_id, sent = line.strip().split('\t')
                            sent = bpe_encoder.decode([
                                int(tok) if tok not in {'<unk>', '<mask>'} else tok
                                for tok in sent.split()
                            ])
                        except:
                            sample_id = line.strip()
                            sent = ''
                        sample_id = sample_id.split('-')[-1]
                        sent = sent.strip()
                        sample.append(sent)
                    elif line[0] == 'T':
                        try:
                            sent = line.strip().split('\t')[1]
                            sent = bpe_encoder.decode([
                                int(tok) if tok not in {'<unk>', '<mask>'} else tok
                                for tok in sent.split()
                            ])
                        except:
                            sent = ''
                        sent = sent.strip()
                        sample.append(sent)
                    elif line[0] == 'H':
                        _, full_logprob, sent = line.strip().split('\t')
                        sample.append(sent)
                        sent = bpe_encoder.decode([
                            int(tok) if tok not in {'<unk>', '<mask>'} else tok
                            for tok in sent.split()
                        ])
                        sent = sent.strip()
                        sample.append(sent)
                    elif line[0] == 'P':
                        sent = line.strip().split('\t')[-1]
                        logprobs = [float(prob) for prob in sent.split()]
                        mean_logprob = sum(logprobs) / len(logprobs)
                        probs = [math.pow(2, prob) for prob in logprobs]
                        mean_probs = math.pow(2, mean_logprob)
                        sample.append(mean_probs)
                        sample.append(mean_logprob)
                        sample.append(probs)
                        sample.append(logprobs)
            if sample:
                all_samples.append((sample_id, sample))
                sample = []

            all_samples = sorted(all_samples, key=lambda x: int(x[0]))
            all_samples = [x[1] for x in all_samples]

            with open(os.path.join(generate_dir, f'{data_split}.target'), 'w') as f:
                for sample in all_samples:
                    f.write(sample[3] + '\n')

            with open(os.path.join(generate_dir, f'{data_split}.bpe.target'), 'w') as f:
                for sample in all_samples:
                    f.write(sample[2] + '\n')

            with open(os.path.join(generate_dir, f'{data_split}.prob'), 'w') as f:
                for sample in all_samples:
                    f.write(' '.join([f'{prob:.6f}' for prob in [sample[4]] + sample[6]]) + '\n')

            with open(os.path.join(generate_dir, f'{data_split}.logprob'), 'w') as f:
                for sample in all_samples:
                    f.write(' '.join([f'{prob:.6f}' for prob in [sample[5]] + sample[7]]) + '\n')


if __name__ == '__main__':
    main()