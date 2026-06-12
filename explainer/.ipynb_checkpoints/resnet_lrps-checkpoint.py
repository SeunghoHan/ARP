import torch
from torch import nn
from copy import deepcopy
import re

from explainer.lrp_utils import layers_lookup, Clone, Add

lookup_table = layers_lookup()

"""
# For resnet=18, resnet-34

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, btnk):
        super().__init__()
        self.btnk_layers = self._disassemble_bottleneck(btnk)
        self.nb_layers = len(self.btnk_layers)

    def _disassemble_bottleneck(self, btnk):
        btnk_layers = nn.ModuleDict()
        
        keys, layers = zip(*btnk.named_children())
        keys, layers = list(keys), list(layers)
        
        btnk_layers['clone'] = Clone()
        btnk_layers[keys[0]] = layers[0]
        btnk_layers[keys[1]] = layers[1]
        btnk_layers['relu1'] = nn.ReLU(inplace=True)
        
        btnk_layers[keys[2]] = layers[2]
        btnk_layers[keys[3]] = layers[3]
        
        if 'downsample' in keys:
            btnk_layers[keys[5]] = layers[5]
        
        btnk_layers['add'] = Add()
        btnk_layers['relu3'] = layers[4]
        
        return btnk_layers
"""    
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, btnk, idx):
        super().__init__()
        self.start_idx = idx
        self.btnk_layers = dict(btnk.named_children())
        self.layer_keys, self.nb_layers = self._init_layer_keys()
        
        self.end_idx = self.start_idx + self.nb_layers - 1
        self.isIncludeDS = False
        self.activations = {}
        self.r_residual = None
        
    def _init_layer_keys(self):
        keys = ['clone', 'conv1', 'bn1', 'relu', 'conv2', 'bn2', 'relu', 'conv3', 'bn3', 'downsample', 'add', 'relu'] \
                if 'downsample' in self.btnk_layers.keys() \
                else ['clone', 'conv1', 'bn1', 'relu', 'conv2', 'bn2', 'relu', 'conv3', 'bn3', 'add', 'relu']
        nb_layers =13 if len(keys) == 12 else 11
        
        return keys, nb_layers
        
    def forward(self, x):
        idx = self.start_idx
        
        for key in self.layer_keys:
            if 'clone' in key:
                self.activations[idx] = x; idx += 1
                x_residual = Clone()(x)
                
            elif 'downsample' in key:
                self.isIncludeDS = True
                ds_layers = list(self.btnk_layers[key])
                self.activations[idx] = x_residual; idx += 1
                x_residual = ds_layers[0](x_residual)
                self.activations[idx] = x_residual; idx += 1
                x_residual = ds_layers[1](x_residual)
            
            elif 'add' in key:
                self.activations[idx] = [x, x_residual]; idx += 1
                x = Add()(x, x_residual)
            
            else:  
                self.activations[idx] = x; idx += 1    
                x = self.btnk_layers[key](x)

        return x
        
    def relevance(self, r, s_idx, t_idx, mode, top_k):
        idx = self.end_idx
        a = None
        
        for key in reversed(self.layer_keys):
            if idx > s_idx:
                if 'downsample' in key: idx -= 2
                else: idx -= 1
                continue 
                
            if 'clone' in key:
                lrp_layer = lookup_table[Clone().__class__.__name__](layer=deepcopy(Clone()), mode=mode, top_k=top_k)
                a = self.activations[idx]
                r = lrp_layer(r1=r, r2=self.r_residual)
            elif 'downsample' in key:
                ds_layers = list(self.btnk_layers[key])
                lrp_layer = lookup_table[ds_layers[1].__class__](layer=deepcopy(ds_layers[1]), mode=mode, top_k=top_k)
                self.r_residual = lrp_layer(a=self.activations[idx].data.requires_grad_(True), r=self.r_residual); idx -= 1
                lrp_layer = lookup_table[ds_layers[0].__class__](layer=deepcopy(ds_layers[0]), mode=mode, top_k=top_k)
                self.r_residual = lrp_layer(a=self.activations[idx].data.requires_grad_(True), r=self.r_residual)
            elif 'add' in key:
                lrp_layer = lookup_table[Add().__class__.__name__](layer=deepcopy(Add()), mode=mode, top_k=top_k)
                a1 = self.activations[idx][0]
                a2 = self.activations[idx][1]
                a = [a1, a2]
                r, self.r_residual = lrp_layer(a1=a1.data.requires_grad_(True), a2=a2.data.requires_grad_(True), r=r)
            else:   
                lrp_layer = lookup_table[self.btnk_layers[key].__class__](layer=deepcopy(self.btnk_layers[key]), mode=mode, top_k=top_k)
                a = self.activations[idx]
                r = lrp_layer(a=a.data.requires_grad_(True), r=r)

            if idx == t_idx: break
            else: idx -= 1
        
        return r, a
        


class ResNet_LRP(nn.Module):
    def __init__(self, name, model, device, input_size, nb_classes, mode='z_plus', top_k=1.0, pt_range='FULL'):
        super().__init__()
        self.name = name
        self.model = model
        self.device = device
        self.input_size = input_size
        self.nb_classes = nb_classes
        self.mode = mode
        self.top_k = top_k
        self.pt_range = pt_range
        
        self.activations = dict()
        
        self.layers, self.check_pt = self._disassemble_model()
        self.shapes = self._init_check_pt_shape()
        
    def _disassemble_model(self):
        resnet_num = int(re.sub(r'[^0-9]', '', self.name))
        
        layers = nn.ModuleDict()
        check_pt = list()
        idx = 0
        
        assert self.model != None, 'Model is None'
                
        for key, module in self.model.named_children():
            if 'layer' in key:
                for btnk in module.children():
                    if self.pt_range == 'FULL': check_pt.append(idx+5) # index of 3x3 conv in each Bottleneck block
                    if resnet_num < 50: layers[str(idx)] = BasicBlock(btnk)
                    else: layers[str(idx)] = Bottleneck(btnk, idx)
                        
                    idx = layers[str(idx)].end_idx + 1 # layers[str(idx)].nb_layers 
                    check_pt.append(idx) 
            else:
                layers[str(idx)] = module; idx += 1
                if 'avgpool' in key: layers[str(idx)] = torch.nn.Flatten(start_dim=1); idx += 1
        
        return layers, check_pt
    
    def _init_check_pt_shape(self):
        x_dump = torch.zeros([1, 3, self.input_size, self.input_size]).to(self.device)
        _ = self.forward(x_dump)
        
        shapes = {}
        for idx in self.check_pt:
            if idx in self.activations.keys():
                shapes[idx] = self.get_activation(idx).shape
            else:
                act_keys = list(self.activations.keys())
                for i, a_key in enumerate(act_keys):
                    if len(act_keys)-1 < i: break
                    if idx > act_keys[i] and idx < act_keys[i+1]:
                        shapes[idx] = self.layers[str(a_key)].activations[idx].shape
                        break
                        
        return shapes
                
    
    def _get_last_relevance(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
        if labels == None:
            labels = torch.argmax(logits, 1) # if labels is None, prediction is replaced by labels
        
        # logits = torch.softmax(logits, dim=-1)
        mask = torch.zeros_like(logits)
        for i, idx in enumerate(labels):
            mask[i, idx] = 1
            
        return logits * mask
    
    def get_activation(self, idx):
        assert len(self.activations) > 0, print('Not find activation dict')
        return self.activations[idx]
    
    @torch.no_grad()
    def forward(self, x):
        if len(self.activations) != 0:
            self.activations = dict()
            
        # save input of each layer 
        for key, layer in self.layers.items():
            if int(key) == 0: self.activations[0] = torch.ones_like(x)
            else: self.activations[int(key)] = x
            x = layer(x)
            
        self.activations[int(key)+1] = x
        
        return x
    
    @torch.no_grad()
    def get_target_act_and_pred(self, x, idx):
        target_act = None
        
        pre_key = 0
        for key, layer in self.layers.items():
            if idx == int(key): 
                target_act = x.clone().detach()
            # elif idx > pre_key and idx < int(key):
            #     break
            x = layer(x)
            # pre_key = int(key)
                
        post_logits = torch.softmax(x, dim=-1)
        post_logits = torch.max(post_logits, 1)
        
        return target_act, post_logits      

    def relevance(self, r, y=None, s_idx=-1, t_idx=-1):
        pred = None
        
        if s_idx == -1:
            last_activation = self.forward(r)
            post_logits = torch.softmax(last_activation, dim=-1)
            pred = torch.max(post_logits, 1) # [0]: max values, [1]: index
        
            r = self._get_last_relevance(logits=last_activation.clone().detach(), 
                                         labels=y)
        
        pre_key = -1
        
        for key, layer in reversed(self.layers.items()):
            if s_idx == -1 or (t_idx == 0 and int(key)<=s_idx) or (s_idx < pre_key and s_idx >= int(key)):
                if 'Bottleneck' in layer.__class__.__name__:
                    r, a = layer.relevance(r=r, s_idx=s_idx, t_idx=t_idx, mode=self.mode, top_k=self.top_k)
                else:
                    lrp_layer = lookup_table[layer.__class__](layer=deepcopy(layer), 
                                                                   mode=self.mode,
                                                                   top_k=self.top_k)
                    a = self.activations[int(key)]
                    r = lrp_layer(a=a.data.requires_grad_(True),
                                  r=r)
                
            if (s_idx+1 == pre_key and t_idx !=0) or t_idx == int(key): 
                break
            else: 
                pre_key = int(key)
        
        return r, a, pred
    
    
class ResNet_CLRP(ResNet_LRP):
    def __init__(self, name, model, device, input_size, nb_classes, mode='z_plus', top_k=1.0, pt_range='FULL'):
        super().__init__(name, model, device, input_size, nb_classes, mode='z_plus', top_k=1.0, pt_range=pt_range)
        
    def _get_last_relevance_T(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
        if labels == None:
            labels = torch.argmax(logits, 1) # if labels is None, prediction is replaced by labels
        
        mask = torch.zeros_like(logits)
        for i, idx in enumerate(labels):
            mask[i, idx] = 1
       
        return logits * mask 
        
    def _get_last_relevance_D(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
        if labels == None:
            labels = torch.argmax(logits, 1) 
        
        mask = torch.ones_like(logits)
        _,  n = logits.shape

        for i, idx in enumerate(labels):
            mask[i, idx] = 0
            logits[i].data *= -1 * (1 / (n-1))

        return logits * mask
    
    def _get_last_relevance(self, logits: torch.tensor, labels: torch.tensor) -> torch.tensor:
        r_T = self._get_last_relevance_T(logits, labels)
        r_D = self._get_last_relevance_D(logits, labels)
        
        return r_T + r_D
    

class ResNet_SGLRP(ResNet_LRP):
    def __init__(self, name, model, device, input_size, nb_classes, mode='z_plus', top_k=1.0, pt_range='FULL'):
        super().__init__(name, model, device, input_size, nb_classes, mode='z_plus', top_k=1.0, pt_range=pt_range)
        
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