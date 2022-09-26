<p align="center">
    <img src="./file/S3PRL-logo.png" width="700"/>
    <br>
    <img src="./file/System_peque.png" width="700"/>
    <br>
</p>

# Voice Disorder Detection Downstream 

In this branch there is a downstream for Automatic Voice Disorder Detection (AVDD) based on S3PRL toolkit. As frontend you can use any Self-Supervised representation from S3PRL including Wavlm, HuBERT, Wav2Vec, etc (see more upstreams at https://github.com/s3prl/s3prl). As backend there are several models available for classification including: 
* *MLP:* Basic pooling + linear layer
* *CNNSelfAttention:* Convolutional Neural Network with Self Attention mechanism
* *Transformer:* 2-layer ViT-Transformer

## Databases 

So far the databases included in this downstream are the following (both are free available):
* **SVD** Saarbruecken Voice Database 
(http://www.stimmdatenbank.coli.uni-saarland.de/help_en.php4)
* **AVFAD** Advanced Voice Function Assessment Database 
(http://acsa.web.ua.pt/AVFAD.htm)

## Usage
For running an experiment you need to clone the repo and install S3PRL toolkit as authors indicated at https://github.com/s3prl/s3prl).

### Installation
1. Clone repo

```
git clone https://github.com/dayanavivolab/s3prl.git 
```

2. Create environment and activate (Python >= 3.6)

``` 
python -m venv /scratch/user/miniconda3/envs/s3prl
``` 

```
source /scratch/user/miniconda3/envs/s3prl/bin/activate
```

3. Install **sox** 
4. Install **s3prl**

```
python -m pip install -e ./
```

5. Install **fairseq**

```
git clone https://github.com/pytorch/fairseq
```

```
cd fairseq
```

```
python -m pip install --editable ./
```

6. Install **torch** (if you already have your own torch skip this step)

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### Run experiment
For running experiments you can use:   
```
sh run_voicedisorder.sh 
```

## Configuration
But first, please configurate the script accordingly by choosing: 
* Training type: finetune, basic
* Audiotype: phrase, aiu, a_n
* Frontend: wavlm, hubert, wav2vec, fft (see more upstreams at https://github.com/s3prl/s3prl)
* Backend: Transformer, CNNSelfAtt, MLP

Also, see the following table with a description of the config files at s3prl/downstream/voicedisorder/config.

<p align="center">
    <img src="./file/S3PRL-Disorder-config.png" width="800"/>
</p>

### Results
Experimental results are located in s3prl/result/downstream/yourfoldername.

Also you can get several system performance metrics by using: 
```
python compute_metrics_full.py s3prl/result/downstream yourdirname 5
```
# 

## Citation

If you find this toolkit useful, please consider citing the following paper.
```
@inproceedings{s3prldisorder_iberspeech2022,
  author={Dayana Ribas and Miguel Angel Pastor and Antonio Miguel and David Martinez and Alfonso Ortega and Eduardo Lleida},
  title={{S3prl-Disorder: Open-Source Voice Disorder Detection System based in the Framework of S3PRL-toolkit.}},
  year=2022,
  booktitle={Proc. Iberspeech 2022}
}
```
