# Voice Disorder Detection Downstream 
In this branch there is a downstream for Automatic Voice Disorder Detection (AVDD) based on S3PRL toolkit. As frontend you can use any Self-Supervised representation from S3PRL including Wavlm, HuBERT, Wav2Vec, etc (see more upstreams at https://github.com/s3prl/s3prl). As backend there are several models available for classification including: 
* *MLP:* Basic pooling + linear layer
* *CNNSelfAttention:* Convolutional Neural Network with Self Attention mechanism
* *Transformer:* 2-layer ViT-Transformer

<p align="center">
    <img src="./file/S3PRL-disorder.png" width="500"/>
    <br>
</p>

## Databases 
So far the databases included in this downstream are the following (both are free available):
* **SVD** Saarbruecken Voice Database 
(http://www.stimmdatenbank.coli.uni-saarland.de/help_en.php4)
* **AVFAD** Advanced Voice Function Assessment Database 
(http://acsa.web.ua.pt/AVFAD.htm)
* **THALENTO** ViVoLab Database for Automatic Detection of Voice Disorders (Under construction, soon release)
(http://dihana.cps.unizar.es/~thalento/)

## Usage
For running an experiment you need to clone the repo and install S3PRL toolkit as authors indicated at https://github.com/s3prl/s3prl).

### Installation
1. Clone repo

```
git clone https://github.com/dayanavivolab/s3prl.git -b voicedisorder
```

2. Create and activate environment (Python >= 3.6)

``` 
cd s3prl
python -m venv s3prl_voicedisorder
source s3prl_voicedisorder/bin/activate
pip install --upgrade pip (optional)
```

3. Install **sox** 

```
apt-get install sox
```
4. Install **s3prl**

```
pip install -e ./
```

5. Install **fairseq**

```
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./
```

6. Install **torch** (if you already have your own torch skip this step) and other packages

```
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboard
```

### Run experiment

1. Locate databases

First you need to locate the datasets in the following directories (see the audio path format in downstream/voicedisorder/data/lst/*.json): 
- downstream/voicedisorder/data/audio/Saarbruecken
- downstream/voicedisorder/data/audio/AVFAD
- downstream/voicedisorder/data/audio/THALENTO

2. Configuration

Select a config file according to your experiment in downstream/voicedisorder/data/config and modify it according to the experiment requirements. 

3. For running experiments to do train and evaluation you can use (modify accordingly):   
```
sh run_voicedisorder.sh 
```

3a. For running only evaluation of a certain list of audios using a pretrained model you can use (modify accordingly):   
```
sh run_voicedisorder_evaluation.sh 
```

## Configuration
But first, please configurate the script accordingly by choosing: 
* Training type: finetune, basic
* Audiotype: phrase, aiu, a_n
* Frontend: wavlm, hubert, wav2vec, fft (see more upstreams at https://github.com/s3prl/s3prl)
* Backend: Transformer, CNNSelfAtt, MLP

|  Parameter    | Value                     | Notes                      |
|---------------|---------------------------|----------------------------|
| **Module: Runner**                                                     |
| total_steps   | numerical value           | Number of epochs/iterations for training the model (ex: 30000)|
| log_step      | numerical value           | Each this number of epochs save a log file (ex: 500)        |
| eval_step     | numerical value           | Each this number of epochs evaluate the model (ex: 100)     |
| save_step     | numerical value           | Each this number of epochs save a checkpoint of the model (ex: 100)   |
| **Module: Optimizer**                                                  |
| name          | TorchOptim                |                            |
| torch_optim_name | Adam                   | Namo of the torch optimizer |
| lr            | numerical value           | Learning rate (Ex: 1.0e-5)  |
| **Module: listrc**                                                     |
| audiotype     | phrase, aiu, a_n          | Type of audio in train/test lists |
| gender        | both, male, female        | Gender of audio in train/test lists |
| traindata     | train_phrase_both_meta_data_SVD | Name of the training list (ex: train_phrase_both_meta_data_SVD_fold1.json) |
| testdata      | test_phrase_both_meta_data_SVD | Name of the training list (ex: test_phrase_both_meta_data_SVD_fold1.json) |
| augment_type  | none, mixup, batch        | Type of method for data augmentation  |
| mixup_alpha   | numerical value [0,1]     |  Parameter for mixup method |
| mixup_beta    | numerical value [0,1]     |  Parameter for mixup method | 
| batch_augment_n | numerical value         |  Number of times the bathc is augmented, one time=1, two times=2, ten times=10 |
| **Module: datarc**                            |
| train_batch_size | numerical value        | Size of the batch in training (ex: 4, 16, 32, 64, 128)|
| eval_batch_size | numerical value         |  Size of the batch in training (ex: 4, 16, 32, 64, 128)|
| num_workers   | numerical value           | Number of workers           |
| **Module: visualrc**                                                  |
| roc           | 0, 1                      | Enable computing ROC curve at each eval_step |
| embeddings    | 0, 1                      | Enable saving embeddings at each eval step |
| **Module: modelrc**                                                   |
| select        | 'Transformer','DeepModel','UtteranceLevel' | Models available for classification |


### Results
Experimental results are located in s3prl/result/downstream/yourfoldername.

Also you can get several system performance metrics by using: 
```
python compute_metrics_full.py s3prl/result/downstream yourdirname 5
```
Or you can visualize the embeddings by using (remember save the embeddings setting embeddings '1' in the config file): 
```
python compute_umap_tsne.py result/downstream/yourdirname/embeddings
```
# 

## Citation

If you find this toolkit useful, please consider citing the following paper.
```
@inproceedings{s3prldisorder_iberspeech2022,
  author={Dayana Ribas and Miguel Angel Pastor and Antonio Miguel and David Martinez and Alfonso Ortega and Eduardo Lleida},
  title={{S3prl-Disorder: Open-Source Voice Disorder Detection System based in the Framework of S3PRL-toolkit.}},
  year=2022,
  booktitle={Proc. Iberspeech 2022},
  pages={136--140},
  url={https://www.isca-speech.org/archive/pdfs/iberspeech_2022/ribas22_iberspeech.pdf}
}
```

## Contact
email: dribas@unizar.es
