import torch
from torch import nn
from copy import deepcopy

from explainer.lrp_utils import layers_lookup
from explainer.lrp import LRP

class SGLRP(LRP):
    def __init__(self, layer_info, mode='z_plus', top_k=1.0):
        super().__init__(layer_info, mode='z_plus', top_k=1.0)
    
    def _get_last_relevance_T(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
        if labels == None:
            labels = torch.argmax(logits, 1) # if labels is None, prediction is replaced by labels
        
        logits = torch.softmax(logits, dim=-1)
        mask = torch.zeros_like(logits)  
        
        for i, idx in enumerate(labels):
            mask[i, idx] = 1
            logits[i] = logits[i] * (1 - logits[i, idx].item())
        
        return logits * mask
        
    def _get_last_relevance_D(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
        if labels == None:
            labels = torch.argmax(logits, 1) 
        
        logits = torch.softmax(logits, dim=-1)
        mask = torch.ones_like(logits)
        
        for i, idx in enumerate(labels):
            mask[i, idx] = 0
            logits[i] = -1 * (logits[i] * logits[i, idx].item())
                                                                                               
        return logits * mask
    
    def _get_last_relevance(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
        r_T = self._get_last_relevance_T(logits, labels)
        r_D = self._get_last_relevance_D(logits, labels)
        
        return r_T + r_D
    