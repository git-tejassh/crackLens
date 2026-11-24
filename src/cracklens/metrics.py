import numpy as np
import torch
import torch.nn as nn

from monai.losses import DiceCELoss

#cutoff above which to consider the value as mask or background
CUTOFF = 0.5
def iou(gt : list[float], pred: list[float]) -> float:
    '''
    Calculate the intersection over union between ground truth and predicted values. 
    The score measures how much the two inputs overlap relative to their combined area.
    
    Params:
    gt: Ground truth values.
    pred: Predicted values.

    Returns:
    The intersection-over-union score.
    ''' 

    cutoff = CUTOFF
    
    gt_bool = gt > 0.5
    probs = torch.sigmoid(pred).squeeze().cpu().numpy()
    pred_bool = probs > CUTOFF
    

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()

    return 0 if union == 0 else intersection/union


    
def precision_recall_f1(gt: list[float], pred: list[float]) -> float:
    ''' 
    Calculate the precision of predicted values against ground truth values. The score measures the proportion of positive predictions that are correct.

    Params:
    gt: Ground truth values.
    pred: Predicted values.

    Returns:
    The precision score.
    '''
    
    gt_bool = gt > 0.5
    probs = torch.sigmoid(pred).squeeze().cpu().numpy()
    pred_bool = probs > CUTOFF
    
    tp = np.sum(pred_bool & gt_bool)
    fp = np.sum(pred_bool & ~gt_bool)
    tn = np.sum(~pred_bool & ~gt_bool)
    fn = np.sum(~pred_bool & gt_bool)
    precision, recall = tp/(tp+fp), tp/(tp+fn)
    return precision, recall, (2*(precision*recall)/(precision + recall))

def diceBCEloss(gt: list[float], pred: list[float]) -> float:
    ''' 
    Calculate the Dice binary cross-entropy loss between ground truth and predicted values. The loss measures prediction accuracy while emphasizing overlap between the two inputs.

    Params:
    gt: Ground truth values.
    pred: Predicted values.

    Returns:
    The Dice binary cross-entropy loss.
    '''
    
    criterion = DiceCELoss(
       sigmoid=True,
       batch=True 
    )

    return criterion(pred, gt)
    


