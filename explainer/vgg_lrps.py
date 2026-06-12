import torch
from torch import nn
from copy import deepcopy
import re

from explainer.lrp_utils import layers_lookup

lookup_table = layers_lookup()

class VGG_LRP(nn.Module):
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
        vgg_num = int(re.sub(r'[^0-9]', '', self.name))
        
        layers = nn.ModuleDict()
        check_pt = list()
        idx = 0
        tmp_pt = -1
        
        assert self.model != None, 'Model is None'
                
        for key, module in self.model.named_children():
            if 'avgpool' in key: 
                layers[str(idx)] = module; idx += 1
                layers[str(idx)] = torch.nn.Flatten(start_dim=1); idx += 1
            else:
                for m in module: 
                    layers[str(idx)] = m; idx += 1
                    if 'CONV' in m.__class__.__name__.upper(): 
                        if self.pt_range == 'FULL': check_pt.append(idx) 
                        else: tmp_pt = idx
                    
                    if 'MAXPOOL' in m.__class__.__name__.upper():
                        if tmp_pt > 0:
                            check_pt.append(tmp_pt) 
                            tmp_pt = -1
                            
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
        
        for key, layer in self.layers.items():
            if idx == int(key): target_act = x.clone().detach()
            x = layer(x)
                
        post_logits = torch.softmax(x, dim=-1)
        post_logits = torch.max(post_logits, 1)
        
        return target_act, post_logits 
        
    def relevance(self, r, y=None, s_idx=-1, t_idx=0):
        pred = None
        
        if s_idx == -1:
            last_activation = self.forward(r)
            post_logits = torch.softmax(last_activation, dim=-1)
            pred = torch.max(post_logits, 1) # [0]: max values, [1]: index
        
            r = self._get_last_relevance(logits=last_activation.clone().detach(), 
                                         labels=y)
    
        pre_key = -1
        for key, layer in reversed(self.layers.items()):
            if s_idx == -1 or s_idx >= int(key):
                lrp_layer = lookup_table[layer.__class__](layer=deepcopy(layer), 
                                                               mode=self.mode,
                                                               top_k=self.top_k)
                a = self.activations[int(key)]
                r = lrp_layer(a=a.data.requires_grad_(True),
                              r=r)
            
            if t_idx == int(key): break
        
        return r, a, pred
    
class VGG_CLRP(VGG_LRP):
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
    

class VGG_SGLRP(VGG_LRP):
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
    
    
    
    
class VGG_SLRP(nn.Module):
    def __init__(self, name, model, device, input_size, nb_classes, mode='z_plus', top_k=1.0, pt_range='FULL'):
        super().__init__()
        self.name = name
        self.model = model
        self.device = device
        self.input_size = input_size
        self.nb_classes = nb_classes
        self.mode = mode
        self.top_k = top_k
        self.pt_range = pt_range='FULL'
        
        self.activations = dict()
        self.gradients = dict()
        self.grad_batch = dict()
        
        self.layers, self.check_pt = self._disassemble_model()
        self.shapes = self._init_check_pt_shape()
        
    def _disassemble_model(self):
        vgg_num = int(re.sub(r'[^0-9]', '', self.name))
        
        layers = nn.ModuleDict()
        check_pt = list()
        idx = 0
        tmp_pt = -1
        
        assert self.model != None, 'Model is None'
                
        for key, module in self.model.named_children():
            if 'avgpool' in key: 
                layers[str(idx)] = module; idx += 1
                layers[str(idx)] = torch.nn.Flatten(start_dim=1); idx += 1
            else:
                for m in module: 
                    layers[str(idx)] = m; idx += 1
                    if 'CONV' in m.__class__.__name__.upper(): 
                        if self.pt_range == 'FULL': check_pt.append(idx) 
                        else: tmp_pt = idx
                    
                    if 'MAXPOOL' in m.__class__.__name__.upper():
                        if tmp_pt > 0:
                            check_pt.append(tmp_pt) 
                            tmp_pt = -1
            
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
    
    def save_gradient(self, idx):
        def hook(grad):
            self.gradients[idx] = grad
        return hook
        
    # @torch.no_grad()
    def forward(self, x):
        if len(self.activations) != 0:
            self.activations = dict()
        
        # save input of each layer 
        for key, layer in self.layers.items():
            if int(key) == 0: self.activations[0] = torch.ones_like(x).data.requires_grad_(True)
            else: 
                self.activations[int(key)] = x
                # if int(key) in self.check_pt: 
                x.register_hook(self.save_gradient(int(key)))
            x = layer(x)
            
        self.activations[int(key)+1] = x
        
        return x
    
    def get_gradients(self, target_cls):
        grads = {}
        for i, t_cls in enumerate(target_cls): 
            t_cls.backward(retain_graph=True)
            for key, grad in self.gradients.items():
                if key in grads.keys(): grads[key] = torch.cat([grads[key], grad[i].unsqueeze(0)], 0)
                else: grads[key] = grad[i].unsqueeze(0)
                    
        return grads
        
    def relevance(self, r, y=None, s_idx=-1, t_idx=0):
        pred = None
        
        if s_idx == -1:
            last_activation = self.forward(r)
            post_logits = torch.softmax(last_activation, dim=-1)
            pred = torch.max(post_logits, 1) # [0]: max values, [1]: index
            self.grad_batch = self.get_gradients(pred[0])
            
            r = self._get_last_relevance(logits=last_activation.clone().detach(), 
                                         labels=y)
    
        pre_key = -1
        for key, layer in reversed(self.layers.items()):
            if s_idx == -1 or s_idx >= int(key):
                lrp_layer = lookup_table[layer.__class__](layer=deepcopy(layer), 
                                                               mode=self.mode,
                                                               top_k=self.top_k)
                a = self.activations[int(key)]
                r = lrp_layer(a=a.data.requires_grad_(True),
                              r=r)
                
                # if a.dim() == 2:
                #     grad = torch.where(self.grad_batch[int(key)] > 0, self.grad_batch[int(key)], 0.)
                #     grad_sum =  torch.sum(grad, 1).unsqueeze(1)
                #     grad /= grad_sum
                #     r *= grad.squeeze()
                # else:
                #     grad_avg_pool = torch.nn.AvgPool2d(self.grad_batch[int(key)].shape[-2:])(self.grad_batch[int(key)])
                #     grad_avg_pool = torch.where(grad_avg_pool > 0, grad_avg_pool, 0.).squeeze()
                #     grad_sum = torch.sum(grad_avg_pool, 1).unsqueeze(1)
                #     grad_avg_pool /= grad_sum
                #     grad_avg_pool = grad_avg_pool.unsqueeze(-1).unsqueeze(-1)
                #     r *= grad_avg_pool
                
            if t_idx == int(key): 
                grad_avg_pool = torch.nn.AvgPool2d(self.grad_batch[int(key)].shape[-2:])(self.grad_batch[int(key)])
                grad_avg_pool = torch.where(grad_avg_pool > 0, grad_avg_pool, 0.).squeeze()
                grad_sum = torch.sum(grad_avg_pool, 1).unsqueeze(1)
                grad_avg_pool /= grad_sum
                grad_avg_pool = grad_avg_pool.unsqueeze(-1).unsqueeze(-1)
                r *= grad_avg_pool
                break
                
        return r, a, pred