from pathlib import Path
import glob
import numpy as np
import pandas as pd

from scipy import interpolate
from expected_cost import utils
from scipy.optimize import brentq
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_curve, auc
from IPython import embed
import matplotlib.pyplot as plt
import torch 


def load_data(DATA_PATH):
    # Loads text files with scores and labels and returns a dataframe
    # Scores are preactivations of a Softmax layer with 2 units
    # The first column are the logits for the class 1 and the second for the class 0
    file_list = glob.glob(DATA_PATH)
    df_list = []
    for file in file_list:
        fold = file.split('/')[-1].split('_')[1]
        df = pd.read_csv(file, delimiter=' ', header=None)
        df['fold'] = fold
        df_list.append(df)
    dff = pd.concat(df_list, axis=0)
    df_final = dff.drop(columns=[2])
    df_final = df_final.rename(columns={0:'logid', 1:'labels', 3:'scores', 4:'h_scores'})
    df_final['labels'] = df_final['labels'].map({'HEALTH':0, 'PATH':1})
    
    return df_final





def softmax_to_logits(x):
    return np.log(x / (1 - x))

def plot_rocs_and_hists(df, outfile):
    scores = np.array(df['scores'])
    labels = np.array(df['labels'])
    # pos es 1 y son patologias porque es lo que se quiere detectar
    pos_scores = df.loc[df['labels'] == 1, 'scores'].values
    neg_scores = df.loc[df['labels'] == 0, 'scores'].values
    all_scores = np.r_[pos_scores, neg_scores]
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1)
    fnr = 1-tpr
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)

    auc_roc = auc(fpr, tpr)

    # Histograma
    h, e = np.histogram(all_scores, bins=50)
    #print(e)
    c = (e[:-1]+e[1:])/2
    hp = np.histogram(pos_scores, bins=e, density=True)[0]
    hn = np.histogram(neg_scores, bins=e, density=True)[0]

    fig, axs = plt.subplots(1, 2, figsize = (15, 10))
    axs = np.atleast_2d(axs)

    ax1 = axs[0,0]
    ax2 = axs[0,1]

    ax1.plot(fpr, tpr, color='blue', label='eer= %.2f, auc= %.2f' % (eer*100, auc_roc*100))
    ax2.plot(c, hp, color='green', label='PATH')
    ax2.plot(c, hn, color='red', linestyle='--', label='HEALTH')


    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    ax1.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax1.set_title('ROC curve')
    ax2.set_title('SCORES DIST')

    fig.tight_layout()
    plt.savefig(outfile)

# hacer una funcion logposteriors from logits
def get_log_posteriors_from_logits(logits):
    exp_logits = np.exp(logits - np.max(logits))
    return exp_logits / np.sum(exp_logits)

