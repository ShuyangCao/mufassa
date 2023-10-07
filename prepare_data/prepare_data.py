import os
import argparse
import regex as re
import spacy
import csv
import shutil
import random
import spacy_alignments
from fairseq.data.encoders.gpt2_bpe_utils import get_encoder
from concurrent.futures import ProcessPoolExecutor


nlp = spacy.load('en_core_web_sm')
TOKEN_RE = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
bpe_encoder = get_encoder(
    os.path.join(os.getenv('EXPDIR'), 'pretrain_language_models', 'fairseq.gpt2', 'encoder.json'),
    os.path.join(os.getenv('EXPDIR'), 'pretrain_language_models', 'fairseq.gpt2', 'vocab.bpe')
)


def process_target(text, src_text):
    text_doc = nlp(text)
    text_tokens = [token.text for token in text_doc]
    text_ent = [token.ent_iob_ for token in text_doc]
    text_pos = [token.pos_ for token in text_doc]

    gpt2_tokens = []
    bpe_tokens = []
    for token in re.findall(TOKEN_RE, text):
        gpt2_tokens.append(token)
        token = "".join(bpe_encoder.byte_encoder[b] for b in token.encode("utf-8"))
        bpe_tokens.append([
            bpe_encoder.encoder[bpe_token] for bpe_token in bpe_encoder.bpe(token).split(" ")
        ])

    src_bpe_tokens = []
    for token in re.findall(TOKEN_RE, src_text):
        token = "".join(bpe_encoder.byte_encoder[b] for b in token.encode("utf-8"))
        src_bpe_tokens.extend([
            bpe_encoder.encoder[bpe_token] for bpe_token in bpe_encoder.bpe(token).split(" ")
        ])

    text2gpt2, gpt22text = spacy_alignments.get_alignments(text_tokens, gpt2_tokens)

    bpe_poss = []
    bpe_starts = []
    for i, gpt2_token in enumerate(gpt2_tokens):
        bpe_poss.extend([text_pos[gpt22text[i][0]]] * len(bpe_tokens[i]))
        bpe_start = ["0"] * len(bpe_tokens[i])
        bpe_start[0] = "1"
        bpe_starts.extend(bpe_start)

    bpe_tokens = [token for bpe_token in bpe_tokens for token in bpe_token]

    assert len(bpe_poss) == len(bpe_starts) == len(bpe_tokens)

    bpe_poss = ' '.join(bpe_poss)
    bpe_starts = ' '.join(bpe_starts)
    bpe_text = " ".join([str(token) for token in bpe_tokens])
    src_bpe_text = " ".join([str(token) for token in src_bpe_tokens])

    return bpe_text, bpe_poss, bpe_starts, src_bpe_text


def build_shuffle(text):
    text = ' ' + text
    tokens = list(re.findall(TOKEN_RE, text))
    random.shuffle(tokens)
    text = ''.join(tokens).strip()
    text = ' '.join(text.strip().split())

    bpe_tokens = []
    for token in re.findall(TOKEN_RE, text):
        token = "".join(bpe_encoder.byte_encoder[b] for b in token.encode("utf-8"))
        bpe_tokens.append([
            bpe_encoder.encoder[bpe_token] for bpe_token in bpe_encoder.bpe(token).split(" ")
        ])
    bpe_text = " ".join([str(token) for token in bpe_tokens])

    return text, bpe_text


def build_empty(text):
    return '', ''


def build_mask_entity(text):
    text_doc = nlp(text)
    text_tokens = [token.text for token in text_doc]
    text_ent = [token.ent_iob_ for token in text_doc]
    text_pos = [token.pos_ for token in text_doc]

    gpt2_tokens = []
    bpe_tokens = []
    for token in re.findall(TOKEN_RE, text):
        gpt2_tokens.append(token)
        token = "".join(bpe_encoder.byte_encoder[b] for b in token.encode("utf-8"))
        bpe_tokens.append([
            bpe_encoder.encoder[bpe_token] for bpe_token in bpe_encoder.bpe(token).split(" ")
        ])

    text2gpt2, gpt22text = spacy_alignments.get_alignments(text_tokens, gpt2_tokens)

    mask_tokens = []
    mask_bpe_tokens = []
    already_mask = set()
    for i, gpt2_token in enumerate(gpt2_tokens):
        if any([text_ent[text_i] == 'B' for text_i in gpt22text[i]]):
            first_text_i = [text_i for text_i in gpt22text[i] if text_ent[text_i] == 'B' and text_i not in already_mask]
            if first_text_i:
                first_text_i = first_text_i[0]
                already_mask.add(first_text_i)
                mask_tokens.append(' <mask> ')
                mask_bpe_tokens.append('<mask>')
        elif any([text_ent[text_i] == 'I' for text_i in gpt22text[i]]):
            continue
        elif any([text_pos[text_i] == 'PROPN' or text_pos[text_i] == 'NUM' for text_i in gpt22text[i]]):
            first_text_i = [text_i for text_i in gpt22text[i] if (text_pos[text_i] == 'PROPN' or text_pos[text_i] == 'NUM') and text_i not in already_mask]
            if first_text_i:
                first_text_i = first_text_i[0]
                already_mask.add(first_text_i)
                mask_tokens.append(' <mask> ')
                mask_bpe_tokens.append('<mask>')
        else:
            mask_tokens.append(gpt2_token)
            mask_bpe_tokens.extend(bpe_tokens[i])

    mask_text = ''.join(mask_tokens)
    mask_text = ' '.join(mask_text.strip().split())

    mask_bpe = ' '.join([str(token) for token in mask_bpe_tokens])
    return mask_text, mask_bpe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_file')
    parser.add_argument('--target_file')
    parser.add_argument('--output_dir')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    original_dir = os.path.join(args.output_dir, 'original')
    mask_dir = os.path.join(args.output_dir, 'mask_entity_propn')
    shuffle_dir = os.path.join(args.output_dir, 'token_shuffle')
    empty_dir = os.path.join(args.output_dir, 'empty')

    os.makedirs(original_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(shuffle_dir, exist_ok=True)
    os.makedirs(empty_dir, exist_ok=True)

    with open(args.source_file) as fsrc, open(args.target_file) as ftgt:
        sources = [' '.join(line.strip().split()) for line in fsrc]
        targets = [' '.join(line.strip().split()) for line in ftgt]

    with open(os.path.join(original_dir, 'test.bpe.source'), 'w') as fbpesrc, \
            open(os.path.join(original_dir, 'test.bpe.target'), 'w') as fbpetgt, \
            open(os.path.join(original_dir, 'test.pos.target'), 'w') as fbpetgtpos, \
            open(os.path.join(original_dir, 'test.bs.target'), 'w') as fbpetgtstart:
        for source, target in zip(sources, targets):
            bpe_target, bpe_target_poss, bpe_target_start, bpe_source = process_target(target, source)

            fbpesrc.write(bpe_source + '\n')
            fbpetgt.write(bpe_target + '\n')
            fbpetgtpos.write(bpe_target_poss + '\n')
            fbpetgtstart.write(bpe_target_start + '\n')

    with open(os.path.join(mask_dir, 'test.source'), 'w') as fsrc, \
            open(os.path.join(mask_dir, 'test.bpe.source'), 'w') as fbpesrc:
        with ProcessPoolExecutor() as executor:
            futures = []
            for source in sources:
                futures.append(executor.submit(build_mask_entity, source))
            results = [future.result() for future in futures]
        for source, source_bpe in results:
            source = ' '.join(source.strip().split())
            fsrc.write(f'{source}\n')
            fbpesrc.write(f'{source_bpe}\n')

    random.seed(0)
    with open(os.path.join(shuffle_dir, 'test.source'), 'w') as fsrc, \
            open(os.path.join(shuffle_dir, 'test.bpe.source'), 'w') as fbpesrc:
        results = []
        for source in sources:
            results.append(build_shuffle(source))
        for source, source_bpe in results:
            source = ' '.join(source.strip().split())
            fsrc.write(f'{source}\n')
            fbpesrc.write(f'{source_bpe}\n')

    with open(os.path.join(empty_dir, 'test.source'), 'w') as fsrc, \
            open(os.path.join(empty_dir, 'test.bpe.source'), 'w') as fbpesrc:
        results = []
        for source in sources:
            results.append(build_empty(source))
        for source, source_bpe in results:
            source = ' '.join(source.strip().split())
            fsrc.write(f'{source}\n')
            fbpesrc.write(f'{source_bpe}\n')

    # copy target to all dirs
    shutil.copyfile(args.target_file, os.path.join(original_dir, 'test.target'))
    shutil.copyfile(args.target_file, os.path.join(mask_dir, 'test.target'))
    shutil.copyfile(args.target_file, os.path.join(shuffle_dir, 'test.target'))
    shutil.copyfile(args.target_file, os.path.join(empty_dir, 'test.target'))

    shutil.copyfile(os.path.join(original_dir, 'test.bpe.target'), os.path.join(mask_dir, 'test.bpe.target'))
    shutil.copyfile(os.path.join(original_dir, 'test.bpe.target'), os.path.join(shuffle_dir, 'test.bpe.target'))
    shutil.copyfile(os.path.join(original_dir, 'test.bpe.target'), os.path.join(empty_dir, 'test.bpe.target'))


if __name__ == '__main__':
    main()
