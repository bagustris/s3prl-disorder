# -*- coding: utf-8 -*- #
"""*********************************************************************************************"""
#   FileName     [ mixup.py ]
#   Author       [ Vivolab, Dayana Ribas ]
#   Copyright    [ Copyleft(c), Vivolab, University of Zaragoza, Spain ]
"""*********************************************************************************************"""
# Mixup data augmentation  

import numpy as np
import torch


def mixup_wavs_samelabel(x, y, alpha=0.1, beta=0.1):
    
    '''Returns mixed inputs pairs of target and lambda. 
    But mixing feature vectors with the same label (for the binary label case), 
    such that features with label 1 are only mixed with features with label 1, and so on.'''

    if alpha > 0:
        lam = np.random.beta(alpha,beta)
    else:
        lam = 1

    #print(y)
    y = np.asarray(y,dtype=int)
    #print(y)
    
    x0 = x[y==0]
    y0 = y[y==0]
    batch_size = x0.size()[0]    
    index0 = torch.randperm(batch_size)
    mixed_x0 = lam * x0 + (1 - lam) * x0[index0,:]
    
    x1 = x[y==1]
    y1 = y[y==1]
    batch_size = x1.size()[0]    
    index1 = torch.randperm(batch_size)
    mixed_x1 = lam * x1 + (1 - lam) * x1[index1,:]
    '''    
    print('Shape x:')
    print(x.shape)
    print('Shape x0:')
    print(x0.shape)
    print('Shape y:')
    print(y.shape)
    print('Shape y0:')
    print(y0.shape)
    print('Shape y1:')
    print(y1.shape)
    print('Lam:')
    print(lam)
    '''
    mixed_x = torch.cat((mixed_x0,mixed_x1))
    y_b = np.concatenate((y0[index0.tolist()],y1[index1.tolist()]))
    index = torch.randperm(mixed_x.size()[0])
    mixed_out = mixed_x[index,:]
    y_out = y_b[index.tolist()]
    '''
    print('Shape mixed_x:')
    print(mixed_x.shape)
    print('y:')
    print(y)
    print('y_b:')
    print(y_b)
    print('y_out:')
    print(y_out)
    '''
    return mixed_out, y_out

    
