#!/bin/bash

# Please modify this accordingly 
python=/mnt/cephfs/home/voz/dribas/software/miniconda3/envs/s3prl/bin/python3
echo $python

audiotype=$1        # phrase, aiu, a_n
upstream=$2         # hubert, wavlm, wav2vec2, etc
test_fold=fold$3    # fold1, fold2, fold3, ...
model=$4            # Transformer, 
ISSUE=$5            # select a name for your experiment 
type=$6             # finetune, basic
seed=$7             # select a number for the seed 

# Compute scores of Calibration set
#name_eval_folder=calibration_29min       # Select a name for the evaluation results
#$python run_downstream.py -m evaluate -u ${upstream} -d voicedisorder -c "downstream/voicedisorder/config/config_${audiotype}_evaluation.yaml" -n dis${type}_${audiotype}_${upstream}_${model}_${ISSUE}_seed${seed}_$test_fold/$name_eval_folder -i result/downstream/$FOLDER/dis${type}_${audiotype}_${upstream}_${model}_${ISSUE}_seed${seed}_$test_fold/dev-best.ckpt -o "config.downstream_expert.datarc.test_fold='$test_fold'"

# Compute scores of Test set
$python run_downstream.py -m evaluate -u ${upstream} -d voicedisorder -c "downstream/voicedisorder/config/config_${audiotype}_evaluation.yaml" -n dis${type}_${audiotype}_${upstream}_${model}_${ISSUE}_seed${seed}_$test_fold -i result/downstream/dis${type}_${audiotype}_${upstream}_${model}_${ISSUE}_seed${seed}_$test_fold/dev-best.ckpt -o "config.downstream_expert.datarc.test_fold='$test_fold'"
