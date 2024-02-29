import torch
import numpy as np


# Data augmentation
def mixup_data(x, y, alpha=0.1, beta=0.1, use_cuda=False):
    '''Returns mixed inputs, pairs of target and lambda '''
    if alpha > 0:
        lam = np.random.beta(alpha,beta)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index,:,:]
    index = index.tolist()
    y = np.asarray(y)
    y_b = y[index]
    
    return mixed_x, y, y_b, lam

def mixup_data_samelabel(x, y, alpha=0.1, beta=0.1, use_cuda=False):
    '''Returns mixed inputs pairs of target and lambda. 
    But mixing feature vectors with the same label (for the binary label case), 
    such that features with label 1 are only mixed with features with label 1, and so on.'''
    if alpha > 0:
        lam = np.random.beta(alpha,beta)
    else:
        lam = 1

    y = np.asarray(y)

    x0 = x[y==0]
    y0 = y[y==0]
    batch_size = x0.size()[0]    
    index0 = torch.randperm(batch_size)
    mixed_x0 = lam * x0 + (1 - lam) * x0[index0,:,:]
    
    x1 = x[y==1]
    y1 = y[y==1]
    batch_size = x1.size()[0]    
    index1 = torch.randperm(batch_size)
    mixed_x1 = lam * x1 + (1 - lam) * x1[index1,:,:]
        
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
    mixed_out = mixed_x[index,:,:]
    y_out = y_b[index.tolist()]
    
    print('Shape mixed_x:')
    print(mixed_x.shape)
    print('y:')
    print(y)
    print('y_b:')
    print(y_b)
    print('y_out:')
    print(y_out)
    
    return mixed_out, y, y_out, lam

def batch_distortion_augment(x, y, n, alpha=0.1, beta=0.1): 
    x_temp = torch.clone(x)
    y_temp = y
    for _ in range(n):
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha, beta)
        x_temp = torch.cat((x_temp, mixed_x))
        y_temp = np.concatenate((y_temp, y_b))    
    '''
    print('Shape batch size:')
    print(x.shape)  
    print('Shape new batch size:')
    print(x_temp.shape)  
    '''
    return x_temp, y_temp

def batch_distortion_augment_samelabel(x, y, n, alpha=0.1, beta=0.1): 
    x_temp = torch.clone(x)
    y_temp = y
    for _ in range(n):
        mixed_x, y_a, y_b, lam = mixup_data_samelabel(x, y, alpha, beta)
        x_temp = torch.cat((x_temp, mixed_x))  #Concat the original with the new mixed versions
        y_temp = np.concatenate((y_temp, y_b))  

    print('Shape batch size:')
    print(x.shape)  
    print('Shape new batch size:')
    print(x_temp.shape)    

    return x_temp, y_temp

def mixup_criterion(criterion, predicted, y_a, y_b, lam):
    return lam * criterion(predicted, y_a) + (1-lam) * criterion(predicted, y_b)