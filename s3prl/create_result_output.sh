#!/bin/bash

direc=~/s3prl/result/downstream/finetune-trainaugment
feat=hubert
model=Transformer

awk '{print $1" "$2}' ${direc}/pathfinetune_augment_phrase_${feat}_${model}_trainAVFAD-SVD_fold1/test_fold1_truth_predict_score.txt > tmp
awk '{print $3}' ${direc}/pathfinetune_augment_phrase_${feat}_${model}_trainAVFAD-SVD_fold1/test_fold1_truth_predict_score.txt > tmp1

awk '{print $1" "$2}' ${direc}/pathfinetune_augment_phrase_${feat}_${model}_trainAVFAD-SVD_fold2/test_fold2_truth_predict_score.txt >> tmp
awk '{print $3}' ${direc}/pathfinetune_augment_phrase_${feat}_${model}_trainAVFAD-SVD_fold2/test_fold2_truth_predict_score.txt >> tmp1

awk '{print $1" "$2}' ${direc}/pathfinetune_augment_phrase_${feat}_${model}_trainAVFAD-SVD_fold3/test_fold3_truth_predict_score.txt >> tmp
awk '{print $3}' ${direc}/pathfinetune_augment_phrase_${feat}_${model}_trainAVFAD-SVD_fold3/test_fold3_truth_predict_score.txt >> tmp1

awk '{print $1" "$2}' ${direc}/pathfinetune_augment_phrase_${feat}_${model}_trainAVFAD-SVD_fold4/test_fold4_truth_predict_score.txt >> tmp
awk '{print $3}' ${direc}/pathfinetune_augment_phrase_${feat}_${model}_trainAVFAD-SVD_fold4/test_fold4_truth_predict_score.txt >> tmp1

awk '{print $1" "$2}' ${direc}/pathfinetune_augment_phrase_${feat}_${model}_trainAVFAD-SVD_fold5/test_fold5_truth_predict_score.txt >> tmp
awk '{print $3}' ${direc}/pathfinetune_augment_phrase_${feat}_${model}_trainAVFAD-SVD_fold5/test_fold5_truth_predict_score.txt >> tmp1


paste -d ' ' tmp tmp1 > ~/s3prl/result/downstream/output_${feat}_${model}.txt
rm tmp
rm tmp1
