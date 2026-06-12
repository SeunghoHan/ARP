import os
import numpy as np
import torch
from torch import nn
from torchvision import models
from torchvision.models import VGG16_Weights

def load_trained_model(model_name, dataset_name, nb_classes):
    """
        Load a target model trained on a dataset such as CIFAR, CUB-200-2011, etc. (except ImageNet)
        
        Args:
            dataset_name: name of the dataset used for training
            model_name: name of the model to load 
            nb_classes: the number of classes for given dataset
        Returns:
            model_ft: trained model 
            input_size: input size of the model
    """
    
    if model_name == "resnet50":
        model_ft = models.resnet50(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, nb_classes)
        input_size = 224

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=True)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, nb_classes)
        input_size = 224

    elif model_name == "vgg16":
        model_ft = models.vgg16(pretrained=True)
        # model_ft = models.vgg16(weights=VGG16_Weights.DEFAULT)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, nb_classes)
        input_size = 224

    elif model_name == "densenet121":
        model_ft = models.densenet121(pretrained=True)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, nb_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()
        
    
    # load finetuning weights for given dataset
    path = os.path.join('/archive/workspace/XAI/research/ICLR/CLAM/trained_models', model_name)
    # path = os.path.join('trained_models', model_name)
    path = os.path.join(path, '{}_best.pth'.format(dataset_name))
    assert os.path.exists(path), print('Trained model is not exist.')
    
    model_ft.load_state_dict(torch.load(path))
    
    return model_ft, input_size
    

def extract_model_info(model, train_dl, device):
    """
        Extract the layer informations from given trained model  
        Args:
            model: target model to extract information from
            train_dl: dataloader 
        Returns:
            info: dictionary variable containing layer information, etc.
    """
    
    model.eval()

    layer_info = {}
    shape_info = {}
    layer_name = {}
    info = {}

    x_batch, _ = next(iter(train_dl))
    x_batch = x_batch.to(device)

    for key, module in model.named_children():
        layers_by_key = nn.ModuleList()
        layer_shape_by_key = []
        l_name_by_key = []
        
        for m in module:
            x_batch = m(x_batch)
            layer_shape_by_key.append(x_batch.shape)
            layers_by_key.append(m)
            l_name_by_key.append(m.__class__.__name__.upper())

        layer_info[key] = layers_by_key
        shape_info[key] = layer_shape_by_key
        layer_name[key] = l_name_by_key

    info['layer' ] = layer_info
    info['shape'] = shape_info
    info['name'] = layer_name

    return info

def extract_vgg16_info(model, train_dl, device):
    """
        Extract the layer informations from given trained model  
        Args:
            model: target model to extract information from
            train_dl: dataloader 
        Returns:
            info: dictionary variable containing layer information, etc.
    """
    
    model.eval()

    layer_info = {}
    shape_info = {}
    layer_name = {}
    info = {}

    x_batch, _ = next(iter(train_dl))
    x_batch = x_batch.to(device)

    for key, module in model.named_children():
        layers_by_key = nn.ModuleList()
        layer_shape_by_key = []
        l_name_by_key = []
        
        if 'avgpool' in key:
            x_batch = module(x_batch)
            layer_shape_by_key.append(x_batch.shape)
            layers_by_key.append(module)
            l_name_by_key.append(module.__class__.__name__.upper())
            
            flatten = torch.nn.Flatten(start_dim=1)
            x_batch = flatten(x_batch)
            layers_by_key.append(flatten)
            l_name_by_key.append(flatten.__class__.__name__.upper())
            # x_batch = x_batch.reshape(x_batch.shape[0], x_batch.shape[1]*x_batch.shape[2]*x_batch.shape[3])
        else:
            for m in module:
                x_batch = m(x_batch)
                layer_shape_by_key.append(x_batch.shape)
                layers_by_key.append(m)
                l_name_by_key.append(m.__class__.__name__.upper())

        layer_info[key] = layers_by_key
        shape_info[key] = layer_shape_by_key
        layer_name[key] = l_name_by_key

    info['layer' ] = layer_info
    info['shape'] = shape_info
    info['name'] = layer_name

    return info


def generate_icp(sz_patches, nb_classes):
    """
        Generate icps with size of patches
        
        Args: 
            sz_patches: Dictionary, key: conv idx, value: size of patch for each conv
        Returns:
            icp_dict: Dictionary for generated icp 
    """
    if type(sz_patches) == dict:
        sz_patches = list(set(sz_patches.values()))
    
    icp_dict = {}
    for sz_p in sz_patches:
        icp = []
        while(len(icp) < nb_classes):
            new_icp = [np.random.randint(0, 2) for i in range(sz_p**2)]
            checker = [i == new_icp for i in icp]
            if True in checker: continue 
            icp.append(new_icp)
        icp_dict[sz_p] = np.array(icp).reshape(nb_classes, sz_p, sz_p)
    
    return icp_dict



def get_xy_and_val_without_duplication(topk_idx, xy, d_map):
    """
        Fine x and y coordinates with minimum dist_val without duplicaiton  
        
        Args:
            topk_idx: List for indice of min_dist_val as ascending order
            xy: List for x and y coordinates of min_dist_val
            d_map: List for attr_dist_map
        
        Returns:
            new_xy: Dictionary for x and y coordinates without duplication; key: attr_idx, value: coordinate
            new_d_map: List for attr_dist_map without duplication;  key: attr_idx, value: dist map
    """
    
    new_xy = {}
    new_d_map = {}
    
    topk_idx_0 = topk_idx.pop(0)
    new_xy[topk_idx_0] = xy[topk_idx_0]
    new_d_map[topk_idx_0] = d_map[topk_idx_0]
    
    for idx in topk_idx:
        hasDup = [True for k in list(new_xy.keys()) \
                      if xy[idx][0] == new_xy[k][0] \
                          and xy[idx][1] == new_xy[k][1]]
        
        if hasDup.count(True) == 0:
            new_xy[idx] = xy[idx]
            new_d_map[idx] = d_map[idx]
    
    return new_xy, new_d_map


def get_valid_labels(y, p, th=0.2):
    """
        Filter prediction labels which are same to ground-truth labels and bigger than threshold
        
        Args: 
            y: Ground-truth labels; (B, nb_classes)
            p: Prediction values and labels; [0]: values, [1]: labels, (B, nb_classes)
            th: Threshold value 
            
        Returns: 
            valid_idx: Filtered labels 
    
    """
    
    threshold = th
    pred_val = p[0]
    pred_idx = p[1]

    valid_idx = [i for i, v in enumerate(pred_val)
                 if v > threshold
                 if pred_idx[i] in y]

    return valid_idx


def transpose_xy(init_xy, init_range, last_conv_idx, layer_info, 
                  shape_info, name_info, input_size=224, output_idx=None):
    """
        Find bbox inforamtion in input layer for target x and y coordinates in last conv layer
        
        Args:
            init_xy: Target coordinates of x and y in last conv layer
            init_range: Patch size of last conv layer (size of init bbox)
            last_conv_idx: Index of last conv layer of target model 
            layer_info: Layer information of feature layers in target model
            shape_info: Shape informaiton of feature layers in target model 
            name_info: Name information of feature layers in target model
            input_size: Size of target image
    """
    x = init_xy[0]
    y = init_xy[1]
    range_x = init_range
    range_y = init_range
    
    for i in reversed(range(len(shape_info))):
        cur_w, cur_h = shape_info[i][2:]
        
        if i==0:
            break
        
        if i == 0:
            next_w, next_h = input_size, input_size
        else:
            next_w, next_h = shape_info[i-1][2:]
            
        if 'CONV' in name_info[i]:
            new_range_x = next_w - cur_w
            new_range_y = next_h - cur_h
            range_x += new_range_x
            range_y += new_range_y

        elif 'POOL' in name_info[i]:
            sz_kernel = layer_info[i].kernel_size
            x *= sz_kernel
            y *= sz_kernel
            range_x *= sz_kernel
            range_y *= sz_kernel
        
        elif 'BOTTLENECK' in name_info[i]:
            ratio = int(next_w / cur_w)
            x *= ratio
            y *= ratio
            range_x *= ratio
            range_y *= ratio
            
    return [x, range_x, y, range_y]