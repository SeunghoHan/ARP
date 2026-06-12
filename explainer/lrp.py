import torch
from torch import nn
from copy import deepcopy

from explainer.lrp_utils import layers_lookup

# Referecne: https://github.com/kaifishr/PyTorchRelevancePropagation/tree/master

class LRP(nn.Module):
    def __init__(self, layer_info, mode='z_plus', top_k=1.0):
        super().__init__()
        
        self.mode = mode
        self.top_k = top_k
        # self.eps = 1.0e-05
        
        self.layer_info = layer_info
        self.layer_info_keys = list(self.layer_info.keys())

        # Create LRP network
        self.lrp_layers = self._create_lrp_model()
        self.lrp_keys = list(self.lrp_layers.keys())
        
    def _create_lrp_model(self) -> torch.nn.ModuleList:
        """
            Method builds the model for layer-wise relevance propagation.
            
            Returns:
                LRP-model as module list.
        """
        lookup_table = layers_lookup()
        lrp_layers = {}
        
        for key in reversed(self.layer_info_keys):
            layers_by_key = self.layer_info[key]
            lrp_layer_by_key = nn.ModuleList()
            
            for layer in reversed(layers_by_key):
                try:
                    lrp_layer_by_key.append(lookup_table[layer.__class__](layer=deepcopy(layer),
                                                                          mode=self.mode,
                                                                          top_k=self.top_k))
                except KeyError:
                    message = f"Layer-wise relevance propagation not implemented for " \
                              f"{layer.__class__.__name__} layer."
                    raise NotImplementedError(messagesage)
            
            lrp_layers[key] = lrp_layer_by_key
            
        return lrp_layers
    
    def _get_last_relevance(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
        if labels == None:
            labels = torch.argmax(logits, 1) # if labels is None, prediction is replaced by labels
        
        mask = torch.zeros_like(logits)
        for i, idx in enumerate(labels):
            mask[i, idx] = 1
       
        return logits * mask
    
    def _get_rmap_before_feature_module(self, 
                                        x: torch.tensor,
                                        y: torch.tensor=None):
        # get all activations from target model on x
        activations = list()
        
        with torch.no_grad():
            activations.append(torch.ones_like(x))
            for key in self.layer_info_keys:
                for layer in self.layer_info[key]:
                    x = layer.forward(x)
                    activations.append(x)
                
        activations = activations[::-1]
        activations = [a.data.requires_grad_(True) for a in activations]
        
        last_activation = activations.pop(0)
        post_logits = torch.softmax(last_activation, dim=-1)
        pred_outputs = torch.max(post_logits, 1) # [0]: max values, [1]: index
        
        r = self._get_last_relevance(logits=last_activation.clone().detach(), 
                                     labels=y)
        
        for lrp_k in self.lrp_keys[:len(self.lrp_keys)-1]:
            print(lrp_k)
            lrp_layers_by_key = self.lrp_layers[lrp_k]
            for lrp_layer in lrp_layers_by_key:
                a = activations.pop(0)
                r = lrp_layer.forward(a, r) 
        
#         # propagate classificaiton module
#         lrp_layers_by_key = self.lrp_layers[self.lrp_keys[0]]
#         for lrp_layer in lrp_layers_by_key:
#             a = activations.pop(0)
#             r = lrp_layer.forward(a, r) 
            
#         # propagate avgpool module
#         lrp_layers_by_key = self.lrp_layers[self.lrp_keys[1]]
#         for lrp_layer in lrp_layers_by_key:
#             a = activations.pop(0)
#             r = lrp_layer.forward(a, r) 
            
        return r, a, pred_outputs, activations
    
    def _get_rmap_from_s_to_t(self, r, activations, s_idx, t_idx):
        lrp_layers_by_key = self.lrp_layers[self.lrp_keys[-1]]
        
        for i, lrp_layer in enumerate(lrp_layers_by_key):
            lrp_to_conv_idx = len(lrp_layers_by_key)-1-i
            
            if s_idx == -1 or lrp_to_conv_idx < s_idx:            
                a = activations.pop(0)
                r = lrp_layer.forward(a, r) 
            
            if lrp_to_conv_idx == t_idx: break
        
        return r, a, activations
    
    def forward(self,
                r: torch.tensor,
                y: torch.tensor=None,
                acts: list=None,
                s_idx: int=-1, 
                t_idx: int=0):
        """
            Propagate relevance from s_idx to t_idx
            
            Args: 
                r: Input, which can be an images or intermediate relevances (N, C, H, W)
                y: Input label which will be used in case s_idx == -1
                acts: Remaining activations to be used for relevance prop 
                s_idx: Start index for relevance propagation
                t_idx: End index for relevance propagation
                
            Returns:
                r: Relevance in last conv layer
                a: Activation of the last conv layer
                p: Prediction list ([0]: predictiin values, [1]: prediction indice) / (B, nb_classes)
                acts: Remaining activations after last conv layer
        """
        p = None
        if s_idx == -1: # first relevance prop 
            r, a, p, acts = self._get_rmap_before_feature_module(r, y)
        r, a, acts = self._get_rmap_from_s_to_t(r, acts, s_idx, t_idx)
        
        return r, a, p, acts