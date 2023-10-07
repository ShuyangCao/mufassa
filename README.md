# MuFaSSA

Code for EACL 2023 Finding paper "Multi-View Source Ablation for Faithful Summarization".

## Requirements

```shell
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
```

### Fairseq

```shell
mkdir lib
cd lib
git clone https://github.com/pytorch/fairseq
cd fairseq
git checkout b5a039c2
pip install --editable ./
```

## Models

Set `EXPDIR` to the directory for storing related models.

```shell
export EXPDIR=<diretory_path>
```

### Download Models

#### Fairseq Files

```shell
mkdir -p $EXPDIR/pretrain_language_models/fairseq.gpt2
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json' -P $EXPDIR/pretrain_language_models/fairseq.gpt2
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe' -P $EXPDIR/pretrain_language_models/fairseq.gpt2
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt' -P $EXPDIR/pretrain_language_models/fairseq.gpt2
sed '50258i\<mask> 0' $EXPDIR/pretrain_language_models/fairseq.gpt2/dict.txt > $EXPDIR/pretrain_language_models/fairseq.gpt2/mask_dict.txt
```

#### Models Trained with Ablation

Download the trained models from [Google Drive](https://drive.google.com/drive/folders/1VtIF_l4eqKYUhRu5RIAxWttpzU5iI5ON?usp=share_link), decompress, and put them in the `EXPDIR` directory.

```shell
tar xvf $EXPDIR/trained_models.tar.gz
mv trained_models $EXPDIR
```

## Using MuFaSSA

To use MuFaSSA, you need to have `source_file` and `target_file` ready. Each line of `source_file` corresponds to a source document and the corresponding line of `target_file` is the summary to be evaluated. 

```shell
./run_mufassa.sh -s <source_file> -t <target_file> -m <model> -o <output_dir>
```

`model` can be chosen from `cnndm` or `xsum`.

`output_dir` is the directory for storing the output files, `mufassa_single.txt` and `mufassa.txt`, 
where each line of `mufassa_single.txt` is the score for a sample and `mufassa.txt` is the average score.

## Citation

```bibtex
@inproceedings{cao-etal-2023-multi,
    title = "Multi-View Source Ablation for Faithful Summarization",
    author = "Cao, Shuyang  and
      Ma, Liang  and
      Lu, Di  and
      Logan IV, Robert L  and
      Tetreault, Joel  and
      Jaimes, Alejandro",
    booktitle = "Findings of the Association for Computational Linguistics: EACL 2023",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-eacl.151",
    doi = "10.18653/v1/2023.findings-eacl.151",
    pages = "2029--2047",
    abstract = "In this paper, we present MuFaSSa (Multi-view Faithfulness Scoring via Source Ablation), a metric for evaluating faithfulness of abstractive summaries, and for guiding training of more faithful summarizers. For evaluation, MuFaSSa employs different strategies (e.g., masking entity mentions) to first remove information from the source document to form multiple ablated views. Then, the faithfulness level of each token in a generated summary is measured by the difference between the token generation probabilities when given the original document and the ablated document as inputs to trained summarizers. For training, MuFaSSa uses a novel word truncation objective that drops unfaithful tokens located by MuFaSSa in both the decoder input and output. Alignments with human-annotated faithfulness labels on AggreFact show that MuFaSSa is comparable to or better than existing metrics built on classifiers or QA models pre-trained on other tasks. In experiments on summarization with XSum and CNN/DailyMail, models trained with word truncation using MuFaSSa outperform competitive methods according to both automatic faithfulness metrics and human assessments.",
}
```
