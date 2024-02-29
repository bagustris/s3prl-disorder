'''
Description: Calibration 

'''
from psrcal import calibration
from psrcal.calibration import calibrate, AffineCalLogLoss
import numpy as np
import torch



def calibrate_exp(df_train, df_test, use_bias=True):
    # Priors
    priors_unif = np.array([0.5, 0.5])

    # Train set
    train_scores1 = df_train['scores'].values
    train_scores2 = df_train['h_scores'].values
    trainscores = np.c_[train_scores1, train_scores2]
    trainscores_tensor = torch.tensor(trainscores)
    train_logposts = torch.nn.functional.log_softmax(trainscores_tensor, dim=1)
    train_labels = df_train['labels'].values
    train_labels = torch.as_tensor(train_labels, dtype=torch.int64)
    
    # Test scores
    test_scores1  = df_test['scores'].values
    test_scores2  = df_test['h_scores'].values
    testscores = np.c_[test_scores1, test_scores2]
    testscores_tensor = torch.tensor(testscores)
    test_logposts = torch.nn.functional.log_softmax(testscores_tensor, dim=1)
    
    cal_scores, cal_params = calibrate(train_logposts, train_labels, test_logposts, AffineCalLogLoss, bias=use_bias, priors=priors_unif, quiet=True)
    cal_scores = cal_scores.detach().numpy()

    return cal_scores, cal_params


