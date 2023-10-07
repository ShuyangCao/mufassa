#!/bin/bash

DATA_PROCESSING_FILE=prepare_data/prepare_data.py
GPT2_DIR=$EXPDIR/pretrain_language_models/fairseq.gpt2
DECODE_PROCESSING_FILE=decode_scripts/process_decode.py
MUFASSA_SCORE_FILE=post_processing/get_mufassa_score.py

while getopts s:t:m:o: flag
do
    case "${flag}" in
        s) SOURCE_FILE=${OPTARG};;
        t) TARGET_FILE=${OPTARG};;
        m) MODEL_NAME=${OPTARG};;
        o) OUTPUT_DIR=${OPTARG};;
		esac
done


# Create a temporary directory to store the data
TMP_DIR=$(mktemp -d)
TMP_ORIGINAL_DATA_DIR=$TMP_DIR/original
TMP_MASK_DATA_DIR=$TMP_DIR/mask_entity_propn
TMP_SHUFFLE_DATA_DIR=$TMP_DIR/token_shuffle
TMP_EMPTY_DATA_DIR=$TMP_DIR/empty

mkdir -p $TMP_ORIGINAL_DATA_DIR
mkdir -p $TMP_MASK_DATA_DIR
mkdir -p $TMP_SHUFFLE_DATA_DIR
mkdir -p $TMP_EMPTY_DATA_DIR

python $DATA_PROCESSING_FILE \
    --source_file $SOURCE_FILE \
    --target_file $TARGET_FILE \
    --output_dir $TMP_DIR


# Create the binarized data
for dir in $TMP_ORIGINAL_DATA_DIR $TMP_MASK_DATA_DIR $TMP_SHUFFLE_DATA_DIR $TMP_EMPTY_DATA_DIR
do
    fairseq-preprocess \
        --source-lang source --target-lang target \
        --testpref $dir/test.bpe \
        --destdir $dir/bin \
        --workers 32 \
        --srcdict $GPT2_DIR/mask_dict.txt \
        --tgtdict $GPT2_DIR/mask_dict.txt
done


# Run model on the data

TMP_ORIGINAL_RESULT_DIR=$TMP_DIR/decode_outputs/original
TMP_MASK_RESULT_DIR=$TMP_DIR/decode_outputs/mask_entity_propn
TMP_SHUFFLE_RESULT_DIR=$TMP_DIR/decode_outputs/token_shuffle
TMP_EMPTY_RESULT_DIR=$TMP_DIR/decode_outputs/empty

fairseq-generate $TMP_ORIGINAL_DATA_DIR/bin \
    --path $EXPDIR/trained_models/bart/$MODEL_NAME/original/checkpoint_best.pt \
    --results-path $TMP_ORIGINAL_RESULT_DIR \
    --task translation \
    --score-reference \
    --batch-size 8 --fp16 \
    --truncate-source

fairseq-generate $TMP_MASK_DATA_DIR/bin \
    --path $EXPDIR/trained_models/bart/$MODEL_NAME/mask_entity_propn/checkpoint_best.pt \
    --results-path $TMP_MASK_RESULT_DIR \
    --task translation \
    --score-reference \
    --batch-size 8 --fp16 \
    --truncate-source

fairseq-generate $TMP_SHUFFLE_DATA_DIR/bin \
    --path $EXPDIR/trained_models/bart/$MODEL_NAME/token_shuffle/checkpoint_best.pt \
    --results-path $TMP_SHUFFLE_RESULT_DIR \
    --task translation \
    --score-reference \
    --batch-size 8 --fp16 \
    --truncate-source

fairseq-generate $TMP_EMPTY_DATA_DIR/bin \
    --path $EXPDIR/trained_models/bart/$MODEL_NAME/empty/checkpoint_best.pt \
    --results-path $TMP_EMPTY_RESULT_DIR \
    --task translation \
    --score-reference \
    --batch-size 8 --fp16 \
    --truncate-source

python $DECODE_PROCESSING_FILE \
    --generate_dir $TMP_ORIGINAL_RESULT_DIR $TMP_MASK_RESULT_DIR $TMP_SHUFFLE_RESULT_DIR $TMP_EMPTY_RESULT_DIR


# Get MuFaSSa score
python $MUFASSA_SCORE_FILE \
    --original $TMP_ORIGINAL_RESULT_DIR/test.prob \
    --mask $TMP_MASK_RESULT_DIR/test.prob \
    --shuffle $TMP_SHUFFLE_RESULT_DIR/test.prob \
    --empty $TMP_EMPTY_RESULT_DIR/test.prob \
    --pos $TMP_ORIGINAL_DATA_DIR/test.pos.target \
    --start $TMP_ORIGINAL_DATA_DIR/test.bs.target \
    --output_dir $OUTPUT_DIR


# Remove the temporary directory
rm -r $TMP_DIR
