<p align="center">
    <img src="./file/S3PRL-logo.png" width="900"/>
    <br>
    <br>
    <a href="./LICENSE.txt"><img alt="Apache License 2.0" src="./file/license.svg" /></a>
    <a href="https://creativecommons.org/licenses/by-nc/4.0/"><img alt="CC_BY_NC License" src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg" /></a>
    <a href="https://github.com/s3prl/s3prl/actions"><img alt="Build" src="https://github.com/allenai/allennlp/workflows/Master/badge.svg?event=push&branch=master"></a>
    <a href="#development-pattern-for-contributors"><img alt="Codecov" src="https://img.shields.io/badge/contributions-welcome-brightgreen.svg"></a>
    <a href="https://github.com/s3prl/s3prl/issues"><img alt="Bitbucket open issues" src="https://img.shields.io/github/issues/s3prl/s3prl"></a>
</p>

# Voice Disorder Detection Downstream 

In this branch there is a downstream for Automatic Voice Disorder Detection (AVDD) based on S3PRL toolkit. as As frontend you can use any Self-Supervised representation from S3PRL. As backend there are several models available for classification including: 
* *MLP:* Basic pooling + linear layer
* *CNNSelfAttention:* Convolutional Neural Network with Self Attention mechanism
* *Transformer:* 2-layer ViT-Transformer

## Databases 

So far the databases included in this downstream are:
* *SVD* Saarbruecken Voice Database
* *AVFAD* Advanced Voice Function Assessment Database 

## Usage
For running an experiment you need to clone the repo and install S3PRL toolkit as authors indicated.

# Installation
1. Clone repo
``` git 
clone https://github.com/dayanavivolab/s3prl.git ```
2. Create environment and activate: **Python** >= 3.6
``` python -m venv /scratch/user/miniconda3/envs/s3prl 
source /scratch/user/miniconda3/envs/s3prl/bin/activate 
```
3. Install **sox** on your OS
4. Install s3prl
```cd s3prl
python -m pip install -e ./
```
5. Install **fairseq**
```git clone https://github.com/pytorch/fairseq
cd fairseq
python -m pip install --editable ./
```
6. Install **torch** (if you already have your own torch skip this step)
```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

# Run experiment
For running experiments you can use:   
```
sh run_voicedisorder.sh 
```
Please configurate the script accordingly by choosing: 
* Training type: finetune, basic
* Audiotype: phrase, aiu, a_n
* Frontend: wavlm, hubert, wav2vec, fft (see more upstreams at https://github.com/s3prl/s3prl)
* Backend: Transformer, CNNSelfAtt, MLP

# Results
Experimental results are located in s3prl/result/downstream/yourfoldername.

Also you can get the system performance metrics by using: 
```
python compute_metrics_full.py s3prl/result/downstream yourdirname 5
```
# 

## Citation

If you find this toolkit useful, please consider citing the following paper.
```
@inproceedings{dribas_iberspeech2022,
  author={Dayana Ribas and Miguel Angel Pastor and Antonio Miguel and David Martinez and Alfonso Ortega and Eduardo Lleida},
  title={{S3prl-Disorder: Open-Source Voice Disorder Detection System based in the Framework of S3PRL-toolkit.}},
  year=2022,
  booktitle={Proc. Iberspeech 2022}
}
```
