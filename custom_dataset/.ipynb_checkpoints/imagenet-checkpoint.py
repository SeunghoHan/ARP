import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import datasets, models, transforms
from torchsummary import summary

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import random
import xml.etree.ElementTree as ET
from PIL import Image
from copy import deepcopy

from utils import *
from custom_dataset import cub

class ImageNet2012(Dataset):
    def __init__(self, root, train=True, correct_only=False, model_name=None, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.correct_only = correct_only
        self.model_name = model_name
        
        self.cls_json = os.path.join(self.root, 'imagenet_class_index.json')
        
        self.idx2label = []
        self.cls2label = {}
        self.cls2idx = {}
        self.label2cls = {}
        self.label2idx = {}
        
        self._get_metadata()
        
        self.classes, self.image_paths, self.image_labels = self._get_img_paths()
        self.bboxes = self._get_bbox_info()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        bbox = self.bboxes[idx]
        label = self.image_labels[idx]

        w, h = image.width, image.height
        
        if self.transform:
            image = self.transform(image)

        _, rs_w, rs_h = image.shape
        
        bbox = self._get_resized_bbox(h, w, rs_h, rs_w, bbox)
        
        return image, label, bbox, path
    
    def get_item_with_name(self, name):
        idx = -1
        
        for i, path in enumerate(self.image_paths):
            if name in path: 
                idx = i
                break
                
        if idx == -1:
            print("Cannot find item for name")
            return None
        else:
            return self.__getitem__(i)
        
    def _get_bbox_info(self):
        bbox_dir = os.path.join(self.root, 'bbox')
        
        if self.train:
            bbox_dir = os.path.join(bbox_dir, 'train')
        else:
            bbox_dir = os.path.join(bbox_dir, 'val')
        
        bboxes = []
        
        for image_path in self.image_paths:
            iamge_name = image_path.split('/')[-1].split('.')[0]
            
            bbox_path = os.path.join(bbox_dir, '{}.xml'.format(iamge_name))
            
            tree = ET.parse(bbox_path)
            xml_root = tree.getroot()
            bndbox = xml_root.find('./object/bndbox')
            
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)
            
            width = xmax-xmin
            height = ymax-ymin
            bboxes.append([xmin, ymin, width, height])
            
        return bboxes
    
    def _get_img_paths(self):
        if self.train:
            path = os.path.join(self.root, 'train')
        else:
            if self.correct_only:
                path = os.path.join(self.root, 'val_correct_{}_10'.format(self.model_name))
            else:
                path = os.path.join(self.root, 'val')

        classes = list(self.cls2label.keys())
        
        paths = []
        labels = []
        for classs in classes:
            cls_path = os.path.join(path, classs)
            for img_name in os.listdir(cls_path):
                paths.append(os.path.join(cls_path, img_name))
                labels.append(self.cls2idx[classs])

        return classes, paths, labels
            
        
    def _get_metadata(self):
        class_idx = json.load(open(self.cls_json))
        
        self.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        self.cls2label = {class_idx[str(k)][0]: class_idx[str(k)][1] for k in range(len(class_idx))}
        self.cls2idx = {class_idx[str(k)][0]: int(idx) for k, idx in enumerate(class_idx.keys())}
        self.label2cls = {class_idx[str(k)][1]: class_idx[str(k)][0] for k, idx in enumerate(class_idx.keys())}
        self.label2idx = {class_idx[str(k)][1]: idx for k, idx in enumerate(class_idx.keys())}
           
    def get_label(self, idx=None, class_name=None):
        if idx == None and class_name == None:
            return None 
        elif idx != None:
            return self.idx2label[idx]
        else:
            return self.cls2label[class_name]
    
    def get_idx(self, class_name=None):
        if class_name == None:
            return None
        else:
            return int(self.cls2idx[class_name])
        
        
    def _get_resized_bbox(self, h, w, rs_h, rs_w, bbox):
        w_ratio, h_ratio = rs_w/w, rs_h/h
        bbox[0] = int(bbox[0]*w_ratio)
        bbox[1] = int(bbox[1]*h_ratio)
        bbox[2] = int(bbox[2]*w_ratio)
        bbox[3] = int(bbox[3]*h_ratio)
        
        return bbox