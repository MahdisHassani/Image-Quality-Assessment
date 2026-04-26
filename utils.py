import numpy as np
from scipy.stats import spearmanr, pearsonr

def compute_metrics(preds, targets):
    preds = np.array(preds)
    targets = np.array(targets)
    
    plcc = pearsonr(preds, targets)[0]
    srcc = spearmanr(preds, targets)[0]
    
    return plcc, srcc