'''
Compute performance metrics for all scores in several folds, 
such that there is not need of averaging among folds.
Also it draws the ROC and DET curve.
'''

import os, sys
import numpy as np
from sklearn.metrics import *
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt

def compute_score_binary(test_lbltrue, test_lblpredict, test_scorepredict, name, figure=True):
    # INPUT: 
    # test_lbltrue: true labels (0 or 1)
    # test_lblpredict: predicted labels (0 or 1)
    # test_scorepredict: predicted scores values
    # name: name of the experiment
    # figure: if True, it draws the ROC and histogram scores

    # OUTPUT: score (has 13 metrics values)  
    # score = 0.acc, 1.acc0, 2.acc1, 3.uar, 4.f1score, 5.recall, 6.precision, 7.auc, 8.eer, 9.tp, 10.tn, 11.fp, 12.fp
    score = []
    score.append(accuracy_score(test_lbltrue, test_lblpredict))
    score.append(accuracy_score(test_lbltrue[test_lbltrue==0], test_lblpredict[test_lbltrue==0]))
    score.append(accuracy_score(test_lbltrue[test_lbltrue==1], test_lblpredict[test_lbltrue==1]))
    score.append(balanced_accuracy_score(test_lbltrue, test_lblpredict))
    score.append(f1_score(test_lbltrue,test_lblpredict))
    score.append(recall_score(test_lbltrue,test_lblpredict)) #Sensitivity = Recall
    score.append(precision_score(test_lbltrue,test_lblpredict))
    fpr, tpr, _ = roc_curve(test_lbltrue, test_scorepredict)
    eer_value=brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    tn, fp, fn, tp = confusion_matrix(test_lbltrue, test_lblpredict).ravel()
    N = tn + fp + fn + tp
    score.append(tn / (tn + fp)) #Specificity
    score.append(roc_auc_score(test_lbltrue, test_scorepredict))
    score.append(eer_value)
    score.append(tp/N * 100)
    score.append(tn/N * 100)
    score.append(fp/N * 100)
    score.append(fn/N * 100) 

    if figure == True:
        plt.figure()
        plt.subplot(121)
        plt.plot(fpr,tpr)
        plt.title('ROC Curve: '+name+' EER='+str(eer_value))
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.grid(True)
        
        # Histogram
        test_scorepredict = np.array(test_scorepredict)
        tar = test_scorepredict[test_lbltrue==1]
        non = test_scorepredict[test_lbltrue==0]

        min_edge = min(np.percentile(non, 10), np.percentile(tar, 10))
        max_edge = max(np.percentile(non, 90), np.percentile(tar, 90))
        x = np.linspace(min_edge, max_edge, 100)

        n_tar, _ = np.histogram(tar, x)
        n_non, _ = np.histogram(non, x)

        plt.subplot(122)
        plt.plot(x[:-1], n_non/max(n_non), color='green')
        plt.plot(x[:-1], n_tar/max(n_tar), color='red', linestyle='--')
        plt.legend(('PATH','HEALTH'))
        plt.grid(True)

        #a = plt.get_current_fig_manager()
        #a.window.state('zoomed')
        plt.savefig(name+'.png')
        #plt.savefig(name+'.svg',format='svg',dpi=1000)
        #plt.close()

        plt.savefig('scores_'+name+'.png')

    return np.array(score)


def main(path, experiment, kfold, current_fold):
    path = os.path.join(path, experiment)
    audio, label, predict, scorepredict = [], [], [], []
    
    for k in range(kfold):
        if current_fold==0:
            temp = '%s_fold%d' % (path, k+1)
            name = 'test_fold%d_truth_predict_score.txt' % (k+1)
        else: 
            temp = '%s_fold%d' % (path, current_fold)
            name = 'test_fold%d_truth_predict_score.txt' % (current_fold)
        file = os.path.join(temp, name)
        with open(file, 'r') as f: data = f.readlines()
        for line in data:
            temp = line.split(' ')

            audio.append(temp[0])
            lab = temp[1]
            if lab == 'HEALTH': label.append(0)
            if lab == 'PATH': label.append(1)
            
            pred = temp[2]
            if pred == 'HEALTH': predict.append(0)
            if pred == 'PATH': predict.append(1)
            scorepredict.append(float(temp[3].rstrip('\n')))
    
    label = np.array(label)
    predict = np.array(predict)
    score = compute_score_binary(label, predict, scorepredict, experiment)
    print('%s %i-Folds: Acc=%0.4f, AccNorm=%0.2f, AccPath=%0.2f, UAR=%0.4f, F1Score=%0.4f, Recall=%0.4f, Precision=%0.4f, Specificity=%0.4f, AUC=%0.4f, EER=%0.4f, TP=%0.4f, TN=%0.4f, FP=%0.4f, FN=%0.4f\n ' % 
            (path, k+1, score[0], score[1], score[2], score[3], score[4], score[5], score[6], score[7], score[8], score[9], score[10], score[11], score[12], score[13]))


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) < 2:
        print('\nDescription: Compute metrics\n')
        print('Usage: compute_metrics_full.py path experiment_name number_of_folds current_fold')
        print('path: path to the experiment')
        print('experiment_name: name of the experiment folder without the ''foldx'' (e.g. path_aiu_batch4_wavlm_AttentivePooling_CNNSelfAtt)')
        print('number_of_folds: number of folds involved in the metrics computation (e.g. 5 for 5-folds cross-validation or 1 for a single experiement)')
        print('current_fold: fold involved in the metrics computation (e.g. 3)')
        print('Example Dir: python compute_metrics_full.py /mnt/cephfs/home/voz/shared/scratch/dribas/s3prl/s3prl/result/downstream/basic path_aiu_batch4_wavlm_AttentivePooling_CNNSelfAtt 5 \n')
        print('Example Dir: python compute_metrics_full.py /mnt/cephfs/home/voz/shared/scratch/dribas/s3prl/s3prl/result/downstream/basic path_aiu_batch4_wavlm_AttentivePooling_CNNSelfAtt 1 3\n')
    else:
        path = args[0]
        experiment_name = args[1]
        number_of_folds = int(args[2])
        current_fold = 0
        threshold = None
        
        if len(args)>3:
            current_fold = int(val)
            
        main(path, experiment_name, number_of_folds, current_fold)
        
        
        
