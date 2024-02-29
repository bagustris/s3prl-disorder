'''
Description: Loss functions for the downstream task of voice disorder detection

ECELoss and LogitNormLoss taken from https://github.com/hongxin001/logitnorm_ood/blob/main/common/loss_function.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)
    
# AUCLoss-----------------------------------------------------------
def step(x):
    with torch.no_grad():
        auc = (torch.sign(x) + 1.) /2.
    return auc

class AUCLoss(nn.Module):
    """ 
    AUC loss implementation for PyTorch
    The sigmoid approximation for AUC differentiabiity is described in:
    V.Mingote et al. (2020) Optimization of the AUC ROC Curve using neural network supervectors for text-dependent speaker verification
    """

    def __init__(self,delta=30):
        super(AUCLoss, self).__init__()
        self.delta = delta

    def forward(self, x, y):
        # x is the raw dnn output (examples,2) pq es binario
        # y is the label (examples,1)
        x_score = x[:,1] - x[:,0]

        #Get indexes for target and non target
        sp_index = ( y == 1 ).nonzero().data.squeeze()
        non_sp_index = (y == 0).nonzero().data.squeeze()

        sp_score = x_score[sp_index]
        non_sp_score = x_score[non_sp_index]

        p =  torch.sigmoid( self.delta * (sp_score.sub(non_sp_score[:,None])))
        a = step(sp_score.sub(non_sp_score[:,None])).mean()

        loss = -1 * p.mean()

        return loss

# BrierLoss----------------------------------------------------------- 
class BrierLoss(nn.Module):
    '''
    Brier loss implementation 
    '''

    def _init_(self):
        super(BrierLoss, self)._init_()

    def forward(self, input, target):
        target_one_hot = torch.zeros_like(input)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        brier_loss = torch.mean(torch.sum(torch.pow(input - target_one_hot, 2), dim=1))
        return brier_loss

# LogitNorm Loss-----------------------------------------------------------
class LogitNormLoss(nn.Module):

    def __init__(self, device, t=1.0):
        super(LogitNormLoss, self).__init__()
        self.device = device
        self.t = t

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.t
        return F.cross_entropy(logit_norm, target)
    
# ECELoss-----------------------------------------------------------
class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels, t=1.0):
        softmaxes = F.softmax(logits/t, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


