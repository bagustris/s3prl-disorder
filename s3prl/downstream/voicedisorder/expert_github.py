import os
import torch
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.distributed import is_initialized
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable

from ..model import *
from .model import *
from .dataset import SaarbrueckenDataset, collate_fn
from .transformer import *

import pickle
import numpy as np

from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import matplotlib.pyplot as plt

# Performance metrics 
def uar(labels_true, labels_predict):
    return balanced_accuracy_score(labels_true, labels_predict)

def auc(labels, scores):
    return roc_auc_score(labels, scores)
    
def eer(labels, scores, figure=0, pathfigure='default'):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer_value=brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    if figure==1:
        plt.figure()
        plt.plot(fpr,tpr)
        plt.title('ROC Curve: '+os.path.basename(pathfigure[:-4])+'EER='+str(eer_value))
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.grid(True)
        plt.savefig(pathfigure)
    return eer_value

def compute_plots(path_file):
    with open(path_file,'r') as fid:
        lines=fid.readlines()
    epoch = [] 
    dev_acc, dev_uar, dev_auc, dev_eer = [], [], [], []
    test_acc, test_uar, test_auc, test_eer = [], [], [], []
    for line in lines:
        a = line.split(' ')
        if a[0] == 'dev': 
            epoch.append(int(a[3][:-1]))
            dev_acc.append(100*float(a[4][4:].split(',')[0]))
            dev_uar.append(100*float(a[5][4:].split(',')[0]))
            dev_auc.append(100*float(a[6][4:].split(',')[0]))
            dev_eer.append(100*float(a[7][4:].split('\n')[0]))
        elif a[0] == 'test': 
            test_acc.append(100*float(a[4][4:].split(',')[0]))
            test_uar.append(100*float(a[5][4:].split(',')[0]))
            test_auc.append(100*float(a[6][4:].split(',')[0]))
            test_eer.append(100*float(a[7][4:].split('\n')[0]))        

    e = len(epoch)
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(epoch,dev_acc,'b')
    plt.plot(epoch,test_acc[0:e],'r')
    plt.title('ACC')
    plt.legend(('dev', 'test'))
    plt.grid(True)
    plt.subplot(2,2,2)
    plt.plot(epoch,dev_uar,'b')
    plt.plot(epoch,test_uar[0:e],'r')
    plt.title('UAR')
    plt.legend(('dev', 'test'))
    plt.grid(True)
    plt.subplot(2,2,3)
    plt.plot(epoch,dev_auc,'b')
    plt.plot(epoch,test_auc[0:e],'r')
    plt.title('AUC')
    plt.legend(('dev', 'test'))
    plt.grid(True)
    plt.subplot(2,2,4)
    plt.plot(epoch,dev_eer,'b')
    plt.plot(epoch,test_eer[0:e],'r')
    plt.title('EER')
    plt.legend(('dev', 'test'))
    plt.grid(True)
    manager = plt.get_current_fig_manager() #fullscreen
    manager.full_screen_toggle()
    pathfigure=os.path.dirname(path_file)+'/results_by_epoch.png'
    plt.savefig(pathfigure)

def compute_plots_loss(path_file):
    with open(path_file,'r') as fid:
        lines=fid.readlines()
    epoch, dev_loss, test_loss = [], [], []
    
    for line in lines:
        a = line.split(' ')
        if a[0] == 'dev': 
            epoch.append(int(a[3][:-1]))
            dev_loss.append(float(a[4][5:].split('\n')[0]))
        elif a[0] == 'test': 
            test_loss.append(float(a[4][5:].split('\n')[0]))
    
    plt.figure()
    plt.plot(epoch,dev_loss,'b')
    plt.plot(epoch[0:len(test_loss)],test_loss,'r')
    plt.title('Loss')
    plt.legend(('dev','test'))
    plt.grid(True)
    
    manager = plt.get_current_fig_manager() #fullscreen
    manager.full_screen_toggle()
    pathfigure=os.path.dirname(path_file)+'/loss_by_epoch.png'
    plt.savefig(pathfigure)

# Data augmentation
def mixup_data(x, y, alpha=0.1, use_cuda=False):
    '''Returns mixed inputs, pairs of target and lambda '''
    if alpha > 0:
        lam = np.random.beta(alpha,alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index,:,:]
    index = index.tolist()
    y = np.asarray(y)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def batch_distortion_augment(x, y, n, alpha=0.1): 
    x_temp = torch.clone(x)
    y_temp = y
    for _ in range(n):
        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha)
        x_temp = torch.cat((x_temp, mixed_x))
        y_temp = np.concatenate((y_temp, y_b))    
    '''
    print('Shape batch size:')
    print(x.shape)  
    print('Shape new batch size:')
    print(x_temp.shape)  
    '''
    return x_temp, y_temp

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)




class DownstreamExpert(nn.Module):
    
    def __init__(self, upstream_dim, downstream_expert, expdir, **kwargs):
        super(DownstreamExpert, self).__init__()
        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert['datarc']
        self.modelrc = downstream_expert['modelrc']
        self.listrc = downstream_expert['listrc']
        self.visualrc = downstream_expert['visualrc']
        print('[Expert] - Model: '+self.modelrc['select'])
        print('[Expert] - Loss: CrossEntropyLoss') #print('[Expert] - Loss: '+self.modelrc['Loss']['loss_type'])
        
        DATA_ROOT = self.datarc['root']
        self.fold = self.datarc.get('test_fold') or kwargs.get("downstream_variant")
        if self.fold is None:
            self.fold = "fold1"

        print(f"[Expert] - using the testing fold: \"{self.fold}\". Ps. Use -o config.downstream_expert.datarc.test_fold=fold2 to change test_fold in config.")

        # Create train/test list (json) path
        augment_list=self.listrc['augment_list']
        if augment_list=='NONE':
            train_path = os.path.join(DATA_ROOT, 'data/lst', 'train_'+self.listrc['audiotype']+'_'+self.listrc['gender']+'_meta_data_'+ self.fold +'.json')
        else:
            train_path = os.path.join(DATA_ROOT, 'data/lst', 'train_'+self.listrc['audiotype']+'_'+self.listrc['gender']+'_augment_meta_data_'+augment_list+'_'+self.fold +'.json')
        print(f'[Expert] - Training path: {train_path}')
        
        if self.listrc['oracle']==1:
            test_path = os.path.join(DATA_ROOT, 'data/lst', 'train_'+self.listrc['audiotype']+'_'+self.listrc['gender']+'_meta_data_'+ self.fold +'.json')
        else:
            test_path = os.path.join(DATA_ROOT, 'data/lst', 'test_'+self.listrc['audiotype']+'_'+self.listrc['gender']+'_meta_data_'+ self.fold +'.json')
        print(f'[Expert] - Testing path: {test_path}')
        
        # Loading train/test dataset 
        dataset = SaarbrueckenDataset(DATA_ROOT, train_path, self.datarc['pre_load'])
        trainlen = int((1 - self.datarc['valid_ratio']) * len(dataset))
        lengths = [trainlen, len(dataset) - trainlen]
        
        torch.manual_seed(0)
        self.train_dataset, self.dev_dataset = random_split(dataset, lengths)

        self.test_dataset = SaarbrueckenDataset(DATA_ROOT, test_path, self.datarc['pre_load'])

        # Loading model
        model_cls = eval(self.modelrc['select'])
        model_conf = self.modelrc.get(self.modelrc['select'], {})
        self.projector = nn.Linear(upstream_dim, self.modelrc['projector_dim'])
        self.model = model_cls(
            input_dim = self.modelrc['projector_dim'],
            output_dim = dataset.class_num,
            **model_conf,
        )
        # Set loss function
        self.objective = nn.CrossEntropyLoss()
        self.expdir = expdir
        self.register_buffer('best_score', torch.zeros(1))

    def get_downstream_name(self):
        return self.fold.replace('fold', 'pathology')

    def _get_train_dataloader(self, dataset):
        sampler = DistributedSampler(dataset) if is_initialized() else None
        return DataLoader(
            dataset, batch_size=self.datarc['train_batch_size'],
            shuffle=(sampler is None), sampler=sampler,
            num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn
        )

    def _get_eval_dataloader(self, dataset):
        return DataLoader(
            dataset, batch_size=self.datarc['eval_batch_size'],
            shuffle=False, num_workers=self.datarc['num_workers'],
            collate_fn=collate_fn
        )

    def get_train_dataloader(self):
        return self._get_train_dataloader(self.train_dataset)

    def get_dev_dataloader(self):
        return self._get_eval_dataloader(self.dev_dataset)

    def get_test_dataloader(self):
        return self._get_eval_dataloader(self.test_dataset)

    # Interface
    def get_dataloader(self, mode):
        return eval(f'self.get_{mode}_dataloader')()

    # Interface
    def forward(self, mode, features, labels, filenames, records, **kwargs):
        device = features[0].device
        features_len = torch.IntTensor([len(feat) for feat in features]).to(device=device)

        features = pad_sequence(features, batch_first=True)
        features = self.projector(features)
        
        # Data augmentation
        alpha = self.listrc['mixup_alpha']
        aug_type = self.listrc['augment_type']
        if aug_type == 'mixup':           
            mixed_features, y_a, y_b, lam = mixup_data(features, labels, alpha)
            mixed_features, y_a, y_b = Variable(mixed_features), Variable(torch.from_numpy(y_a).cuda()), Variable(torch.from_numpy(y_b).cuda())
            predicted, features_pooled = self.model(mixed_features, features_len)
            loss = mixup_criterion(self.objective, predicted, y_a, y_b, lam)
            labels = y_a
        elif aug_type == 'batch':
            batchN = np.max((1,self.listrc['batch_augment_n']))  # Ensure a minimum value of 1, duplicate the batch
            features, labels = batch_distortion_augment(features, labels, batchN)
            labels = torch.LongTensor(labels).to(features.device)
            predicted, features_pooled = self.model(features, features_len)
            loss = self.objective(predicted, labels)
        # -------------------------------------------------------
        else:
            predicted, features_pooled = self.model(features, features_len) # Meaning: predicted=logit, features_pooled=embedding
            labels = torch.LongTensor(labels).to(features.device)
            loss = self.objective(predicted, labels)

        
        predicted_classid = predicted.max(dim=-1).indices
        records['acc'] += (predicted_classid == labels).view(-1).cpu().float().tolist()
        records['loss'].append(loss.item())

        records["filename"] += filenames
        records["predict"] += [self.test_dataset.idx2emotion[idx] for idx in predicted_classid.cpu().tolist()]
        records["truth"] += [self.test_dataset.idx2emotion[idx] for idx in labels.cpu().tolist()]

        # Write logit 
        m = nn.Softmax(dim=1)
        score = m(predicted)
        records['score'] += score
        records['score_0'] += score[:,0].view(-1).cpu().float().tolist()
        records['score_1'] += score[:,1].view(-1).cpu().float().tolist()

        # Write embeddings
        #if mode=='test' and features_pooled!=None and self.visualrc['embeddings']==1:
        values = records['acc']
        average = torch.FloatTensor(values).mean().item()
        best = 0
        if average > self.best_score: best=1
        
        if best==1 and features_pooled!=None and self.visualrc['embeddings']==1:
            emb_path = self.expdir + '/embeddings/' + mode
            if not os.path.exists(emb_path): os.makedirs(emb_path)
            for i in range(0,len(filenames)):
                with open(emb_path+'/'+filenames[i]+'.pkl','wb') as fid:
                    data = []
                    data.append(np.asscalar(labels[i].cpu().detach().numpy()))
                    data.append(features[i].cpu().detach().numpy())
                    data.append(features_pooled[i].cpu().detach().numpy())
                    pickle.dump(data, fid, protocol=pickle.HIGHEST_PROTOCOL)
        return loss

    # interface
    def log_records(self, mode, records, logger, global_step, **kwargs):
                               
        lab, lab_predict = [], []
        for i in range(0,len(records["truth"])):
            if records["truth"][i] == 'PATH': lab.append(1)
            else: lab.append(0)
            if records["predict"][i] == 'PATH': lab_predict.append(1)
            else: lab_predict.append(0)
        uar_value = uar(lab, lab_predict)
        auc_value = auc(lab, records["score_1"])
        if mode=='test' and self.visualrc['roc']==1:
            fig_path = self.expdir + '/figures'
            if not os.path.exists(fig_path): os.mkdir(fig_path)
            eer_value = eer(lab, records["score_1"], figure=1, pathfigure=fig_path+'/ROC_step_'+str(global_step)+'.png')
        else:
            eer_value = eer(lab, records["score_1"])
        #--------------------------------------
        
        save_names = []
        for key in ["acc", "loss"]:
            values = records[key]
            average = torch.FloatTensor(values).mean().item()
            logger.add_scalar(
                f'pathologies-{self.fold}/{mode}-{key}',
                average,
                global_step=global_step
            )
            # Write log with ACC, AUC, EER
            with open(Path(self.expdir) / "log_acc_auc.log", 'a') as f:
                if key == 'acc':
                    print(mode+'ACC='+str(average)+' UAR='+str(uar_value)+' AUC='+str(auc_value)+' EER='+str(eer_value))
                    f.write(f'{mode} at step {global_step}: {key}={average}, UAR={uar_value}, AUC={auc_value}, EER={eer_value}\n')
                    if mode == 'dev' and average > self.best_score:
                        self.best_score = torch.ones(1) * average
                        f.write(f'New best on {mode} at step {global_step}: ACC={average}, UAR={uar_value}, AUC={auc_value}, EER={eer_value}\n')
                        save_names.append(f'{mode}-best.ckpt')
            if mode == 'test': compute_plots(self.expdir+"/log_acc_auc.log") 

            # Write log with Loss
            with open(Path(self.expdir) / "log_loss.log", 'a') as f:
                if key == 'loss':
                    print(mode+' Loss='+str(average))
                    f.write(f'{mode} at step {global_step}: {key}={average}\n')
                    if mode == 'test': compute_plots_loss(self.expdir+"/log_loss.log")  
            #------------------------------------------------------ 
            #------------------------------------------------------

        if mode in ["dev", "test"]:
            with open(Path(self.expdir) / f"{mode}_{self.fold}_predict.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["predict"])]
                file.writelines(line)

            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth.txt", "w") as file:
                line = [f"{f} {e}\n" for f, e in zip(records["filename"], records["truth"])]
                file.writelines(line)

            # Write score (logit)
            with open(Path(self.expdir) / f"{mode}_{self.fold}_truth_predict_score.txt", "w") as file:
                line = [f"{f} {t} {p} {s}\n" for f, t, p, s in zip(records["filename"], records["truth"], records["predict"], records["score_1"])]
                file.writelines(line)
            #---------------------------------

        return save_names