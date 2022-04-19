# CLEAE

# Introduction
This repository was used in our paper:  

**Translation-Based Implicit Annotation Projection for Zero-Shot Cross-Lingual Event Argument Extraction**

## Requirements
- pytorch >= 0.4.0
- numpy >= 1.13.3
- sklearn
- python 3.6+
- transformers

## Pretrained Models
Download mBERT-base model from [huggingface](https://huggingface.co/facebook/bert-base-multilingual-cased) and put it in the './mBERT' directory.

Download M2M100-1.2B model from [huggingface](https://huggingface.co/facebook/m2m100_1.2B) and use it to translate event mentions.

## Data Preparation
* Download the [ACE 2005](https://catalog.ldc.upenn.edu/LDC2006T06) dataset.

* Translate the event mentions into target languages with the M2M100 model.

* Organize the data into the format defined in [dataset.py](/dataset.py).


## Training
* Train with command, optional arguments could be found in [main.py](/main.py).

For instance, you can train a model of ZH->EN setting by the following command:
```bash
python -W error::FutureWarning main.py --cuda 0 --hidden_alignment --model 4losspos --lang zh-en --OT_threshold 0.5
```





