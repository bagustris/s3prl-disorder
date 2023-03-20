#!/bin/bash

python=/mnt/cephfs/home/voz/dribas/software/miniconda3/envs/s3prl/bin/python3
echo $python

audiotype=$1
upstream=$2
test_fold=fold$3
model=$4
ISSUE=$5 
type=$6
seed=$7

$python run_downstream.py -n dis${type}_${audiotype}_${upstream}_${model}_${ISSUE}_seed${seed}_$test_fold -m train -u ${upstream} -d voicedisorder -c "downstream/voicedisorder/config/config_${audiotype}_finetune_${model}.yaml" -o "config.downstream_expert.datarc.test_fold='$test_fold'" -f -l -1 -a --seed ${seed} 
$python run_downstream.py -m evaluate -e result/downstream/dis${type}_${audiotype}_${upstream}_${model}_${ISSUE}_seed${seed}_$test_fold/dev-best.ckpt

