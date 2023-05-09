#!/bin/bash

# Please modify this accordingly 
python=/mnt/cephfs/home/voz/dribas/software/miniconda3/envs/s3prl/bin/python3

#Inputs
type=basic   # finetune, basic
audiotype=$1    # phrase, aiu, a_n
upstream=$2     # wavlm, hubert, fft, opensmile
model=$3        # Transformer, CNNSelfAtt, MLP

folds="1 2 3 4 5"
for test_fold in folds; do
    
    # TRAIN basic 
    if [[ "$type" == "basic" ]]
    then
        $python run_downstream.py -n dis${type}_${audiotype}_${upstream}_${model}_${test_fold} -m train -u ${upstream} -d voicedisorder -c "downstream/voicedisorder/config/config_"${audiotype}"_"${model}".yaml" -o "config.downstream_expert.datarc.test_fold='${test_fold}'" -a
    fi
    
    # TRAIN with finetune
    if [[ "$type" == "finetune" ]]
    then
        $python run_downstream.py -n dis${type}_${audiotype}_${upstream}_${model}_$test_fold -m train -u ${upstream} -d voicedisorder -c "downstream/voicedisorder/config/config_${audiotype}_finetune_${model}.yaml" -o "config.downstream_expert.datarc.test_fold='$test_fold'" -f -l -1 -a 
    fi
    
    # EVALUATION
    $python run_downstream.py -m evaluate -e result/downstream/dis${type}_${audiotype}_${upstream}_${model}_$test_fold/dev-best.ckpt
#one
