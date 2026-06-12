import torch
import torch.nn.functional as F
from torch import nn
import torchvision
from torchvision import datasets, models, transforms
from torchsummary import summary

import gc
import re
import time
import math
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np 
import os

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname('__file__'))))

import setting
from utils import *
from custom_dataset import cub
from explainer import vgg_lrps
from explainer.lrp_utils import *

# Class-wise Layer-wise Attribute Model 
class CLAM(nn.Module):
    def __init__(self, 
                 model, 
                 device, 
                 icp_dict,
                 model_name: str=None, 
                 explainer_name: str=setting.NAME_OF_EXPLAINER_LRP,
                 layer_sep_type: str='FULL', 
                 nb_classes: int=0):
        super().__init__()
        
        self.model = model
        self.model.eval()
        self.device = device
        self.icp_dict = icp_dict
        self.model_name = model_name
        self.explainer_name = explainer_name
        self.layer_sep_type = layer_sep_type
        self.nb_classes = nb_classes
        
        self.explainer = self._init_explainer(exp_name=self.explainer_name)
        
        self.conv_idx, self.cla_layers, self.nb_attrs, self.sz_patches, \
            self.sz_attrs, self.lr, self.max_epoches, self.strength = self._init_cla_layers()

        self.nb_epochs = self.max_epoches * len(self.cla_layers.keys())
        
    def _init_explainer(self, exp_name):
        if exp_name == setting.NAME_OF_EXPLAINER_LRP:
            if self.model_name == 'vgg16':
                explainer = vgg_lrps.VGG_LRP(name=self.model_name, model=self.model, 
                                             device=self.device, input_size=input_size, 
                                             nb_classes=self.nb_classes, pt_range=self.layer_sep_type)
            elif self.model_name == 'resnet50':
                print('load ResNet_LRP')
                explainer = resnet_lrps.ResNet_LRP(name=self.model_name, model=self.model, 
                                                   device=self.device, input_size=input_size, 
                                                   nb_classes=self.nb_classes, pt_range=self.layer_sep_type)
            
        elif exp_name == setting.NAME_OF_EXPLAINER_CLRP:
            if self.model_name == 'vgg16':
                explainer = vgg_lrps.VGG_CLRP(name=self.model_name, model=self.model, 
                                             device=self.device, input_size=input_size, 
                                             nb_classes=self.nb_classes, pt_range=self.layer_sep_type)
            elif self.model_name == 'resnet50':
                print('load ResNet_LRP')
                explainer = resnet_lrps.ResNet_CLRP(name=self.model_name, model=self.model, 
                                                    device=self.device, input_size=input_size, 
                                                    nb_classes=self.nb_classes, pt_range=self.layer_sep_type)
            
        elif exp_name == setting.NAME_OF_EXPLAINER_SGLRP:
            if self.model_name == 'vgg16':
                explainer = vgg_lrps.VGG_SGLRP(name=self.model_name, model=self.model, 
                                             device=self.device, input_size=input_size, 
                                             nb_classes=self.nb_classes, pt_range=self.layer_sep_type)
            elif self.model_name == 'resnet50':
                print('load ResNet_LRP')
                explainer = resnet_lrps.ResNet_SGLRP(name=self.model_name, model=self.model,
                                                     device=self.device, input_size=input_size, 
                                                     nb_classes=self.nb_classes, pt_range=self.layer_sep_type)
            
        else:
            print("Invalid explainer name, exiting...")
            exit()
            
        return explainer
        
    def _init_cla_layers(self):
        cla_layers = {}
        
        conv_idx = self.explainer.check_pt
        conv_shapes = self.explainer.shapes

        # max_epoches = setting.max_epoches
        max_epoches = 5
        nb_attrs, sz_patches, sz_attrs, \
            strength, alpha, beta, lr = load_hyperparameters(conv_idx=conv_idx, 
                                                             conv_shape=conv_shapes, 
                                                             model_name=self.model_name, 
                                                             layer_sep_type=self.layer_sep_type)
        
        for i, idx in enumerate(conv_idx):
            cla_layers[idx] = CLA_Layer(device=self.device, 
                                        idx=idx, 
                                        layer_shape=conv_shapes[idx], 
                                        nb_classes=self.nb_classes, 
                                        nb_attrs=nb_attrs[idx],
                                        sz_patch=sz_patches[idx], 
                                        sz_attr=sz_attrs[idx], 
                                        max_epoches=max_epoches, 
                                        alpha=alpha[idx], 
                                        beta=beta[idx], 
                                        lr=lr[idx])

        return conv_idx, cla_layers, nb_attrs, sz_patches, sz_attrs, lr, max_epoches, strength
    
    def _get_icp_as_ts(self, sz_p): 
        # (nb_classes, 1, sz_p, sz_p)
        return torch.tensor(self.icp_dict[sz_p])[:, None, :, :].cuda()
    
    def _get_rand_xy_exclusive_of_t_list(self, t_list, max_val):
        rand = np.zeros(t_list.shape, dtype=int)
        
        for k, l in enumerate(t_list): 
            nb_list_items = len(l)
            while(True):
                rand_xy = [random.randint(0, max_val) for i in range(nb_list_items)]
                checker = [rand_xy[j] == l[j]  for j in range(nb_list_items)]
                if True not in checker: 
                    rand[k] = rand_xy
                    break 
                    
        return rand
    
    def _get_r_max_idx(self, r, idx):
        r_2d = r.max(dim=1)[0]
        r_2d = F.avg_pool2d(input=r_2d, 
                            kernel_size=(self.sz_patches[idx], 
                                         self.sz_patches[idx]), 
                            stride=1)
        
        b, rw, rh = r_2d.shape
        k = self.nb_attrs[idx]
        r_1d = r_2d.reshape([b, rw*rh]) #.clone().detach()
        
        r_topk_val, r_topk_idx = torch.topk(r_1d, k=k, dim=1)
        
        # r_topk_idx -> row, col 나눔
        r_topk_row = np.zeros(r_topk_idx.shape).astype(np.int32)
        r_topk_col = np.zeros(r_topk_idx.shape).astype(np.int32)
        for i, idx in enumerate(r_topk_idx):
            r_topk_row[i] = idx.cpu().numpy()//rw
            r_topk_col[i] = idx.cpu().numpy()%rw
        
        return r_topk_row, r_topk_col
    
    def _generate_pos_neg(self, r, a, y, idx):
        """
            Generate pos/neg inputs for training cla_layers
            
            Args:
                r: Relevance (B, C, W, H)
                a: Activation (B, C, W, H)
                y: Label (B, nb_classes)
                idx: Current conv idx
                
           Returns: 
               f_pos: Pos sub feature (nb_attrs, B, C, sz_patch, sz_patch)
               f_neg: Neg sub feature (nb_attrs, B, C, sz_patch, sz_patch)
               r_topk_row: Top-k row index of relevance (k == nb_attr)
               r_topk_col: Top-k col index of relevance (k == nb_attr)
               
        """
        r_topk_row, r_topk_col = self._get_r_max_idx(r, idx)
        
        r2d_max_idx = r.shape[-1] - self.sz_patches[idx]
        r_row_rand_idx = self._get_rand_xy_exclusive_of_t_list(t_list=r_topk_row, 
                                                               max_val=r2d_max_idx)
        r_col_rand_idx = self._get_rand_xy_exclusive_of_t_list(t_list=r_topk_col,
                                                               max_val=r2d_max_idx)
        
        # (nb_attr, b, d, p, p)
        f_pos = torch.zeros([self.nb_attrs[idx], a.shape[0], a.shape[1], 
                             self.sz_patches[idx], self.sz_patches[idx]]).to(device)
        f_neg = torch.zeros([self.nb_attrs[idx], a.shape[0], a.shape[1], 
                             self.sz_patches[idx], self.sz_patches[idx]]).to(device) 
        
        r_topk_row = r_topk_row.T
        r_topk_col = r_topk_col.T
        r_row_rand_idx = r_row_rand_idx.T
        r_col_rand_idx = r_col_rand_idx.T
        
        
        # Extract sub_feature using top-k index for pos and rand index for neg
        for i in range(self.nb_attrs[idx]):
            indices = []
            for j, a_b in enumerate(a):
                f_pos[i,j] = a_b[:, r_topk_row[i][j]:r_topk_row[i][j]+self.sz_patches[idx],
                             r_topk_col[i][j]:r_topk_col[i][j]+self.sz_patches[idx]]
                f_neg[i,j] = a_b[:, r_row_rand_idx[i][j]:r_row_rand_idx[i][j]+self.sz_patches[idx],
                             r_col_rand_idx[i][j]:r_col_rand_idx[i][j]+self.sz_patches[idx]]
        
        # Add icp to last channel in pos/neg
        icp = torch.stack([self._get_icp_as_ts(self.sz_patches[idx]) 
                           for i in range(self.nb_attrs[idx])], dim=0)
        
        f_pos = torch.cat((f_pos, icp[:, y, :]), dim=2)
        
        rand_p = self._get_rand_xy_exclusive_of_t_list(t_list=y.unsqueeze(1), 
                                                       max_val=self.nb_classes-1)
        rand_p = torch.from_numpy(rand_p).squeeze()
        f_neg = torch.cat((f_neg, icp[:, rand_p, :]), dim=2)
        
        return f_pos, f_neg, r_topk_row, r_topk_col
    
    def _get_mask_for_strength(self, idx, mode='train'):
        if mode == 'train':
            s = self.strength[idx]
        else:
            if len(self.conv_idx) < 10:
                s = 1.2
            else: 
                s = self.strength[idx]-0.05 # for vgg 
            
        p = self.sz_patches[idx] 
        mask = np.zeros([p*2-1, p*2-1])
        w, h = mask.shape
        cp_x, cp_y = w//2, h//2
        
        for i in range(w):
            for j in range(h):        
                dis_x = np.abs(cp_x-i)
                dis_y = np.abs(cp_y-j)
                max_dis = dis_x if dis_x-dis_y>0 else dis_y

                if dis_x==0 and dis_y==0:
                    mask[i, j] = s
                else:
                    for k in range(1, p):
                        if max_dis==k:
                            str_v = s*(0.95**k)
                            if str_v < 1.0:
                                str_v = 1.0
                            mask[i, j] = str_v
                            break
        
        return torch.from_numpy(mask).to(device)
    
    def _get_new_pos(self, base_pos, w, h, idx, mask):
        m_w, m_h = mask.shape
        
        if base_pos-self.sz_patches[idx]+1 < 0: 
            s_pt = 0
            m_s_pt = -(base_pos-self.sz_patches[idx]+1)
        else:
            s_pt = base_pos-self.sz_patches[idx]+1
            m_s_pt = 0
        
        if base_pos+self.sz_patches[idx] > w:
            e_pt = w
            m_e_pt = w - base_pos + (m_w-self.sz_patches[idx])
        else: 
            e_pt = base_pos+self.sz_patches[idx]
            m_e_pt = m_w

        return s_pt, e_pt, m_s_pt, m_e_pt
    
    def _apply_min_dist_to_r(self, r, xy, idx, r_idx=None, mode='train'):
        mask = self._get_mask_for_strength(idx, mode) # (p-a+1, p-a+1)
        
        _, _, w, h = r.shape # r: (B, C, W, H)
        for i, r_b in enumerate(r):
            if r_idx != None:
                r_idx_topk = r_idx[i]  # k==nb_attrs[idx]
            else:
                r_idx_topk = [[0, 0] for i in range(self.nb_attrs[idx])]
                
            for j in range(xy.shape[1]):
                r_x, r_y = r_idx_topk[j]
                d_x, d_y = xy[i, j]
                
                x_start, x_end, mx_s_pt, mx_e_pt = self._get_new_pos(r_x+d_x, w, h, idx, mask.clone().detach())
                y_start, y_end, my_s_pt, my_e_pt = self._get_new_pos(r_y+d_y, w, h, idx, mask.clone().detach())

                r_b[:, x_start:x_end, y_start:y_end] *= mask[mx_s_pt:mx_e_pt, my_s_pt:my_e_pt]
                        
        return r
    
    
    def _combine_r_and_dist(self, r, dist_list, idx, r_topk_row, r_topk_col):
        """
            Generate new relevance map with dist_list 
        
        """
        sz_batch = r.shape[0]
        min_dist_xy = np.zeros([sz_batch, self.nb_attrs[idx], 2], dtype=int) # 2: row, col val
        for i, dist in enumerate(dist_list):
            b, a, w, h = dist.shape  # (b, nb_attrs, p-a+1, p-a+1)
            dist = dist.reshape(b, a, w*h)
            min_dist_pos = dist.argmin(dim=-1)
            min_dist_pos = min_dist_pos[:, i]
            for j, pos in enumerate(min_dist_pos):
                min_dist_xy[j, i, 0] = pos//w
                min_dist_xy[j, i, 1] = pos%h
        
        r_topk_row = r_topk_row.T
        r_topk_col = r_topk_col.T
        
        r_idx = []
        for i in range(sz_batch):
            r_idx_b = [[r_topk_row[i, j], r_topk_col[i, j]] for j in range(self.nb_attrs[idx])]
            r_idx.append(r_idx_b)
         
        return self._apply_min_dist_to_r(r, min_dist_xy, idx, r_idx)
            
    def _get_pre_r_and_a(self, x, y, target_idx):
        r = None
        dist_pos_list = None
        pred = None
        
        s_idx = -1
        t_idx = 0
        
        for idx in reversed(self.conv_idx):
            if r == None: r = x
            else: s_idx = t_idx-1
            t_idx = self.cla_layers[idx].idx
            
            r, a, p  = self.explainer.relevance(r=r, y=y, s_idx=s_idx, t_idx=t_idx)
            if pred == None: pred = p
                
            if idx == target_idx: break
            else:
                with torch.no_grad():
                    f_pos, _, r_topk_row, r_topk_col = self._generate_pos_neg(r, a, y, idx)
                    dist_pos_list = [self.cla_layers[idx](f_p) for f_p in f_pos]
                    r = self._combine_r_and_dist(r, dist_pos_list, idx, r_topk_row, r_topk_col)
                    
        return r, a, pred
    
    def _get_attribute_mask(self, xy, r_shape, idx):
        """
            For test
            # Generate a mask for min_dist point used for relevance refinement 
        """
        b, _, w, h = r_shape
        
        m = torch.ones([b, 1, w, h]).to(device)
        # if idx in self.conv_idx:
        m = self._apply_min_dist_to_r(r=m, xy=xy, idx=idx, mode='test')
        m -= 1.0
        
        return m.clone().detach().cpu()
    
    def get_origal_rs(self, x, y=None, target_idx=-1, return_all=False):
        x = x.to(self.device)
        if y is not None: y = y.to(self.device)
        
        r = None
        rest_a = None
        rs = []
        
        s_idx = -1
        t_idx = 0
        
        # for propagating to input layer, add idx -1
        
        conv_idx_expand = self.conv_idx.copy()  
        if target_idx == -1:
            conv_idx_expand.insert(0, -1) # for input layer 
        
        for idx in reversed(conv_idx_expand):
            if r == None: r = x
            else: s_idx = t_idx-1
            
            if idx == -1: t_idx = 0
            else: t_idx = self.cla_layers[idx].idx
            
            r, a, _, = self.explainer.relevance(r=r, y=y, s_idx=s_idx, t_idx=t_idx)
            if return_all:
                rs.insert(0, r.clone().detach().cpu())
            
            if idx == target_idx: break
        
        if return_all:
            return rs
        else:
            return r
                
    def train(self, start_attr_idx=-1, target_idx=0):
        keys = self.conv_idx
        print('keys: ', keys)
        
        # === for test
        if start_attr_idx >= 0:
            keys = keys[:start_attr_idx+1]
        print('target_idx: ', target_idx)
        
        train_log_path = os.path.join(log_model_data_path, 'ckpt_{}/log.txt'.format(self.explainer_name))
        print('train_log_path: ', train_log_path)
        with open(train_log_path, 'w') as f:
            f.write('')
        
        dist_pos_neg_diff = {k: 0. for k in keys}
        max_stopping_cnt = 1
        
        for i, idx in enumerate(reversed(keys)):
            print('idx: ', idx)
            attr_d, attr_w, attr_h = self.cla_layers[idx].attr_shape[1:]
            min_diff_mean = attr_d * attr_w * attr_h * self.nb_attrs[idx]
            early_stopping_cnt = 0
            
            for epoch in range(self.max_epoches):
                
                dist_pos_neg_diff[idx] = 0.
                
                for step, (x_batch, y_batch) in enumerate(train_dl): 
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    
                    r, a, pred = self._get_pre_r_and_a(x=x_batch, 
                                                       y=y_batch,
                                                       target_idx=idx)
                    
                    f_pos, f_neg, _, _ = self._generate_pos_neg(r=r,
                                                                a=a,
                                                                y=y_batch, 
                                                                idx=idx)
                    
                    
                    loss, pos_mean, neg_mean = self.cla_layers[idx].train(y=y_batch, 
                                                                          p=pred, 
                                                                          f_pos_list=f_pos,
                                                                          f_neg_list=f_neg)
                    
                    dist_pos_neg_diff[idx] += pos_mean.item() - neg_mean.item()
                    
                    
                    # if step % int(steps_of_train/2) == 0 and step > 0:
                    if step % int(steps_of_train/100) == 0 and step > 0:
                        p_output = 'step: {} in epoch {} (idx: {})\n'.format(step, epoch, idx)
                        p_output += 'lr: {}\n'.format(self.cla_layers[idx].opt.param_groups[0]['lr'])
                        p_output += '\tloss {} \n\tpos_mean {:.4f} (amass: {:.4f}) \n\tneg_mean {:.4f}\n'.format(loss.item(), 
                                                                                                                 pos_mean.item(),
                                                                                                                 dist_pos_neg_diff[idx]/step,
                                                                                                                 neg_mean.item())
                        print(p_output)
                        with open(train_log_path, 'a') as f:
                            f.write(p_output)  
                            
                cur_idx_dist_diff_mean = dist_pos_neg_diff[idx] / steps_of_train
                if cur_idx_dist_diff_mean < min_diff_mean:
                    min_diff_mean = cur_idx_dist_diff_mean
                    early_stopping_cnt = 0

                    print('Save model!')
                    torch.save({'epoch': epoch,
                                'layer_state_dict': self.cla_layers[idx].state_dict(),
                                'optstate_dict': self.cla_layers[idx].opt.state_dict(),
                                'dist_pos': dist_pos_neg_diff[idx] / step
                               }, '{}/ckpt_{}/{}.pth'.format(log_model_data_path, 
                                                             self.explainer_name,
                                                             idx))
                    self.cla_layers[idx].scheduler.step()

                else: 
                    early_stopping_cnt += 1

                e_output = self._print_output_epoch(epoch, dist_pos_neg_diff)
                # if early_stopping_cnt == setting.max_stopping_cnt: 
                if early_stopping_cnt == max_stopping_cnt: 
                    e_output += 'Skip to next epoch \n'
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                else:
                    e_output += 'Current stopping_cnt: {}\n\n'.format(early_stopping_cnt)
                print('e_output: ', e_output)
                with open(train_log_path, 'a') as f:
                    f.write(e_output)
                    
                # if early_stopping_cnt == setting.max_stopping_cnt: 
                if early_stopping_cnt == max_stopping_cnt: 
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                    
            if idx == target_idx: break
                
        print("FINISH")
        
    def get_mask(self, x, y=None, target_idx=-1, return_all=False):
        """
            Propagate CLA-layers on x batch
            
            Args: 
                x: Input image batch; (B, C, W, H)
                y: Label; (B, nb_classes), If label is None, prediction will be used as label 
                target_idx: Target conv idx for stopping propagation
            Returns:
                r: Generated relevance combining with mask predicted by attributes
                r_orig: Original relevance on input 
        """
        x = x.to(self.device)
        if y is not None: y = y.to(self.device)

        r = None
        pred = None
        r_attr = []
        acts = []
        attr_xy = []
        attr_map = []
        
        s_idx = -1
        t_idx = 0
        
        # for propagating to input layer, add idx -1
        conv_idx_expand = self.conv_idx.copy()
        if target_idx == -1:
            conv_idx_expand.insert(0, -1)
        
        for idx in reversed(conv_idx_expand):
            if r == None: r = x.clone().detach()
            else: s_idx = t_idx-1
            
            if idx == -1: t_idx = 0
            else: t_idx = self.cla_layers[idx].idx
            
            r, a, p = self.explainer.relevance(r=r, y=y, s_idx=s_idx, t_idx=t_idx)
            
            if pred == None: pred = p
            
            if idx == -1: 
                if return_all:
                    r_attr.insert(0, r.clone().detach().cpu())
                    attr_xy.insert(0, 0)
                    attr_map.insert(0, 0)
                    acts.insert(0, a.clone().detach().cpu())
                break
                
            else:            
                with torch.no_grad():
                    attr_dist_xy, attr_dist_map = self.cla_layers[idx].predict(a=a,
                                                                               p=pred[1], 
                                                                               y=y, 
                                                                               icp=self._get_icp_as_ts(self.sz_patches[idx]))
                    
                    r = self._apply_min_dist_to_r(r, attr_dist_xy, idx, mode='test')
                    
                    if return_all:
                        r_attr.insert(0, r.clone().detach().cpu())
                        attr_xy.insert(0, attr_dist_xy)
                        attr_map.insert(0, attr_dist_map)
                        acts.insert(0, a.clone().detach().cpu())
            
            gc.collect()
            torch.cuda.empty_cache()

            if idx == target_idx: break
        
        if return_all:
            return r_attr, attr_xy, attr_map, acts, pred
        else:
            return r, None, None, None, pred
        
    def push_attributes(self, trainset_dl, x, y=None, target_idx=-1):
        """
            Push each attribute in each layer to the nearest patch in the training set
            
            Args:
                training_dl: Training dataloader; 
                x: Test image; (C, W, H)
                y: Test label; 
                
            Returns:
                outputs: Return output list for given test images with dictionary structure as below 
                    outputs = [
                        {
                            'x': (first test image)
                            'y': (label for first test image) (not neccesary)
                            'p': (prediction for image)
                            '(conv_idx)': {
                                'test_r': relevance map for test image
                                'test_xy': {
                                    '(attr_idx)': (coordinates for current attribute with min_dist), 
                                    '(attr_idx)': (coordinates for current attribute with min_dist), 
                                    ...
                                }
                                'train_img': {
                                    '(attr_idx)': (train image similar to given min_dist_map of test image),
                                    '(attr_idx)': (train image similar to given min_dist_map of test image),
                                    ...
                                }
                                'train_xy': {
                                    '(attr_idx)': (coordinates of min_dist_map similar to min_dist_map of test image),
                                    '(attr_idx)': (coordinates of min_dist_map similar to min_dist_map of test image),
                                    ...
                                }
                            }
                        }
                    ]
        """
        if x.dim() < 4:
            x = x.unsqueeze(0)
        x = x.to(device)
        if y!=None: y = y.to(device) 
        r_attr, attr_xy, attr_dist_map, _, pred = self.get_mask(x=x,
                                                                y=y,
                                                                target_idx=target_idx, 
                                                                return_all=True)
        nb_t_layers = len(r_attr)
        conv_idx_pointer = -1
        pred = pred[1]
        
        ###
        outputs = {}
        outputs['x'] = x.clone().detach().cpu()
        if y!=None: outputs['y'] = y.item()
        outputs['p'] = pred.item()
        ###
    
        for i in reversed(range(nb_t_layers)):
            cur_conv_idx = self.conv_idx[conv_idx_pointer]
            conv_idx_pointer -= 1
            # if cur_conv_idx != target_idx: continue
            print("\tcur_conv_idx: ", cur_conv_idx)

            attr_xy_i = attr_xy[i].squeeze() # (nb_attrs, 2)
            attr_dist_map_i = attr_dist_map[i].squeeze()  # (nb_attrs, p-a+1, p-a+1)

            _, topk_idx = torch.topk(input=attr_dist_map_i.mean(dim=[1,2]), 
                                     k=attr_dist_map_i.mean(dim=[1,2]).shape[0], 
                                     dim=0)
            topk_idx = torch.flip(topk_idx, dims=[0]).tolist()

            attr_xy_dict, attr_dist_map_dict = get_xy_and_val_without_duplication(topk_idx=topk_idx,
                                                                                  xy=attr_xy_i,
                                                                                  d_map=attr_dist_map_i)
    
            min_l2d_img, min_l2d_xy = self._find_nearest_patches_in_trainset(trainset_dl=trainset_dl,
                                                                             dist_map_dict=attr_dist_map_dict,
                                                                             target_idx=cur_conv_idx, 
                                                                             test_pred=pred.item())
            if min_l2d_img==None and min_l2d_xy==None:
                continue
            else:
            ###
                outputs_i_info = {}
                outputs_i_info['test_r'] = r_attr.pop(i).clone().detach().cpu()
                outputs_i_info['test_xy'] = attr_xy_dict
                outputs_i_info['train_img'] = min_l2d_img
                outputs_i_info['train_xy'] = min_l2d_xy
                outputs[cur_conv_idx] = outputs_i_info
                ###

            gc.collect()
            torch.cuda.empty_cache()
            
            if cur_conv_idx == target_idx: break

        # outputs.append(outputs_b)

        return outputs
    
    def _find_nearest_patches_in_trainset(self, trainset_dl, dist_map_dict, target_idx, test_pred):
        """
            Find nearest patches from all train images for given test dist_map by attributes 
            
            Args: 
                trainset_dl: Dataloader of trainset
                dist_map_dict: Distance maps for the attributes of test image (no overlapping coordinates)
                target_idx: Index of target conv layer 
                test_pred: Prediction for test image; [0]: prediction values, [1]: predicted classes 
                
            Returns:
                
        """
        
        min_l2d = {k:sum(dist_map_dict.values()).mean()**3 for k in dist_map_dict.keys()}
        min_l2d_img = {}
        min_l2d_xy = {}
        
        for step, (x_batch, y_batch) in enumerate(trainset_dl): 
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            target_a, train_pred = self.explainer.get_target_act_and_pred(x=x_batch,
                                                                          idx=target_idx)
            if target_a == None:
                return None, None
            
            train_valid_idx = get_valid_labels(y=y_batch, p=train_pred)
            x_batch = x_batch[train_valid_idx]
            y_batch = y_batch[train_valid_idx]
            target_a = target_a[train_valid_idx]
            train_vals = train_pred[0][train_valid_idx]
            train_labels = train_pred[1][train_valid_idx]
            
            # (B, nb_attrs, nb_sub_features, p-a+1, p-a+1)
            with torch.no_grad():
                all_dist_maps = self.cla_layers[target_idx]._get_all_distances_for_subfeatures(a=target_a, 
                                                                                               p=train_labels, 
                                                                                               y=None,
                                                                                               icp=self._get_icp_as_ts(self.sz_patches[target_idx]))
            
                all_dist_maps = all_dist_maps.detach().cpu()
                
                start_min = time.time()
                # number of subfeatures = dw*dh
                dw = target_a.shape[2] - self.sz_patches[target_idx] + 1
                dh = target_a.shape[3] - self.sz_patches[target_idx] + 1
                map_w, map_h = next(iter(dist_map_dict.values())).shape  # (p-a+1, p-a+1)
                for k in dist_map_dict.keys():
                    test_dist_map = dist_map_dict[k] 
                    test_dist_map = test_dist_map.reshape([1, map_w*map_h])

                    for b, dits_maps_b in enumerate(all_dist_maps):
                        if test_pred !=  y_batch[b].item(): continue # consider same class between train and test prediction
                        dist_maps_attr = dits_maps_b[k, :] # consider same attribute index between train and test 

                        for i, d_map_sf in enumerate(dist_maps_attr): 
                            old_min_l2d = min_l2d[k]
                            new_min_l2d = torch.cdist(test_dist_map, 
                                                      d_map_sf.reshape([1, map_w*map_h]).type(torch.DoubleTensor),
                                                      p=2)
                            
                            # get minimum l2-distance between train and test dist_map 
                            if new_min_l2d < old_min_l2d:
                                min_l2d[k] = new_min_l2d
                                min_l2d_img[k] = x_batch[b].clone().detach().cpu()
                                min_l2d_xy[k] = [i//dw, i%dh]
                                
                              
            gc.collect()
            torch.cuda.empty_cache()
            
        return min_l2d_img, min_l2d_xy
    
    def _print_output_epoch(self, e, val_dict):
        it_output = '\n---------------------------------\nFinish epoch: {} \n'.format(e)
        total = 0.
        for k in val_dict.keys():
            dp_mean = val_dict[k] / steps_of_train
            total += dp_mean
            it_output += '\tkey: {}, mean: {:.4f}\n'.format(k, dp_mean)

        it_output += '\tTotal mean: {:.4f}\n'.format(total/len(val_dict.keys()))
        it_output += '\n---------------------------------\n'
        
        return it_output 
   