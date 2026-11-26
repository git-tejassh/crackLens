import numpy as np
import torch
import torch.nn as nn
from skimage.morphology import skeletonize, binary_dilation, disk

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
    

def centerline_iou(gt: list[float], pred: list[float], smooth: float = 1e-6, tolerance = 4) -> float:
    ''' 
    Calculate a connectivity-aware intersection-over-union score for ground truth and predicted values. The score measures their overlap while accounting for the connectivity of the structures.
    It measures the overlap between ground truth centerline and predicted centerline, with a tolerance of 4px otherwise the skeleton is only 1px wide.

    Params:
    gt: Ground truth values.
    pred: Predicted values.
    smooth: A small value used to stabilize the score when overlap is near zero.
    tolerance: is the pixel tolerance allowed

    Returns:
    The connectivity-aware intersection-over-union score.
    '''
    gt_bool = gt.astype(bool)
    pred_bool = pred.astype(bool)
    
    #extracting one pixel wide skeletons
    skel_gt = skeletonize(gt.astype(bool))
    skel_pred = skeletonize(pred.astype(bool))
    if tolerance>0:
        selem = disk(tolerance)
        gt_tol = binary_dilation(gt_bool)
        pred_tol = binary_dilation(pred_bool)
        
    
    #Topologu Precision & Sensitivity
    t_prec = (np.sum(skel_gt & gt_tol) + smooth) / (np.sum(skel_pred) + smooth)
    t_sens = (np.sum(skel_gt & pred_tol)) / (np.sum(skel_gt)+ smooth)
    
    #centerline IOU
    cl_iou = (t_prec * t_sens) / (t_prec + t_sens - t_prec*t_sens)
    return float(cl_iou)
    