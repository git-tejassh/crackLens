import numpy as np
import torch
import torch.nn as nn
from skimage.morphology import skeletonize, binary_dilation, disk

from monai.losses import DiceCELoss


def _check_inputs(gt, pred):
    ''' 
    Validate that two masks are compatible for comparison. The inputs must be two-dimensional Boolean NumPy arrays with matching shapes.

    Params:
    gt: Ground truth mask.
    pred: Predicted mask.

    Returns:
    None.

    Raises:
    TypeError: If either input is not a two-dimensional Boolean NumPy array.
    ValueError: If the input masks have different shapes.
    '''
    
    for m in (gt, pred):       # check both inputs, one at a time
        if not isinstance(m, np.ndarray) or m.dtype != bool or m.ndim!=2:
            raise TypeError('masks must be 2D numpy boolean arrays')
    if gt.shape != pred.shape:  #both masks must be same size
        raise ValueError(f'shape mismatch: {gt.shape} vs {pred.shape}')

#cutoff above which to consider the value as mask or background
CUTOFF = 0.5
def iou(gt_bool: np.ndarray, pred_bool: np.ndarray) -> float:
    '''
    Calculate the intersection over union between ground truth and predicted values. 
    The score measures how much the two inputs overlap relative to their combined area.
    
    Params:
    gt: Ground truth values.
    pred: Predicted values.

    Returns:
    The intersection-over-union score.
    ''' 

    

    _check_inputs(gt_bool, pred_bool)


    intersection = np.logical_and(gt_bool, pred_bool).sum()
    union = np.logical_or(gt_bool, pred_bool).sum()

    return float('nan') if union == 0 else intersection/union


    
def precision_recall_f1(gt_bool: np.ndarray, pred_bool: np.ndarray) -> float:
    ''' 
    Calculate the precision of predicted values against ground truth values. The score measures the proportion of positive predictions that are correct.

    Params:
    gt: Ground truth values.
    pred: Predicted values.

    Returns:
    The precision score.
    '''
    
    _check_inputs(gt_bool, pred_bool)
    
    tp = np.sum(pred_bool & gt_bool)
    fp = np.sum(pred_bool & ~gt_bool)
    tn = np.sum(~pred_bool & ~gt_bool)
    fn = np.sum(~pred_bool & gt_bool)
    precision, recall = tp/(tp+fp), tp/(tp+fn)
    denom = 2*tp + fp + fn
    f1 = 2*tp / denom if denom else float('nan')
    return precision, recall, f1

### TO BE SHIFTED TO LOSSES.PY
# def diceBCEloss(gt: list[float], pred: list[float]) -> float:
#     ''' 
#     Calculate the Dice binary cross-entropy loss between ground truth and predicted values. The loss measures prediction accuracy while emphasizing overlap between the two inputs.

#     Params:
#     gt: Ground truth values.
#     pred: Predicted values.

#     Returns:
#     The Dice binary cross-entropy loss.
#     '''
    
#     criterion = DiceCELoss(
#        sigmoid=True,
#        batch=True 
#     )

#     return criterion(pred, gt)
    

def centerline_iou(gt_bool: np.ndarray, pred_bool: np.ndarray, smooth: float = 1e-6, tolerance = 4) -> float:
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

    _check_inputs(gt_bool, pred_bool)
    
    #extracting one pixel wide skeletons
    skel_gt = skeletonize(gt.astype(bool))
    skel_pred = skeletonize(pred.astype(bool))
    
    
    footprint = disk(tolerance)
    gt_tol = binary_dilation(skel_gt, footprint)
    pred_tol = binary_dilation(skel_pred, footprint)
        
        
    #Topology Precision & Sensitivity
    t_prec = (np.sum(skel_pred & gt_tol) + smooth) / (np.sum(skel_pred) + smooth)
    t_sens = (np.sum(skel_gt & pred_tol)) / (np.sum(skel_gt)+ smooth)
    
    #centerline IOU
    cl_iou = (t_prec * t_sens) / (t_prec + t_sens - (t_prec*t_sens))
    return float(cl_iou)
    
    
    