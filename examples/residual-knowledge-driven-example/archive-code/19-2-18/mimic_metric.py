import torch
import torchvision
from sklearn.metrics import roc_auc_score
import numpy as np

def AUCAccuracy(outputs, labels):
    '''Compute the AUC and accuracy for each sample.
        Convert tensor to numpy for computation
        Args:
            outputs (Variable): predictions, one row per sample (after sigmoid)
            labels (Variable): ground truth labels, one row per sample
        Returns:
            a tensor of floats, one per sample
    '''
    p_np = outputs
    # print ('p_np: \n', p_np)
    y_np = labels
    # print ("y_np shape: ", y_np.shape)       
    pred_p_np = (p_np > 0.5).astype(np.float32)
 
    # accuracy = tensor.from_numpy((pred_x_np == y_np).astype(np.float32)) # still a matrix
    accuracy = ((pred_p_np == y_np).astype(np.float32)) # still a matrix
    accuracy = np.sum(accuracy)/(accuracy.shape[0]*accuracy.shape[1])
    if y_np.shape[0] < 1000: # too small number, no need for AUC
        macro_auc = 0.0
        micro_auc = 0.0
    else:
        macro_auc = roc_auc_score(y_np.astype(np.int32), p_np, average='macro')
        micro_auc = roc_auc_score(y_np.astype(np.int32), p_np, average='micro')
    return [accuracy, macro_auc, micro_auc]
