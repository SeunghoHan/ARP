import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url

class Cub2011(Dataset):
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, correct_only=False, model_name=None, 
                 transform=None, loader=default_loader, download=False):
        self.root = os.path.expanduser(root)
        self.base_folder = '/archive/workspace/datasets/CUB_200_2011/images/'
        self.transform = transform
        self.loader = default_loader
        self.train = train
        self.correct_only = correct_only
        self.model_name = model_name

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.correct_only:
            self.correct_paths = self._get_correct_path()
            self._remove_wrong()
        
    def _remove_wrong(self):
        wrong_idx = []
        for row in self.data.itertuples():
            file_name = row[2]
            file_name = file_name.split('.jpg')[0]
            
            is_wrong = True
            for cor_path in self.correct_paths:
                cor_path = cor_path.split('.jpg')[0]
                if file_name in cor_path:
                    is_wrong = False
                    break
            
            if is_wrong: 
                wrong_idx.append(row[0])
        
        for idx in wrong_idx:
            self.data = self.data.drop(idx, axis=0)
            
    
    def _get_correct_path(self):
        correct_dir = os.path.join(self.root, 'split/val_correct_{}'.format(self.model_name))
        
        if os.path.exists(correct_dir) == False:
            print("Not found correct directory")
            return

        paths = []
        cls_list = os.listdir(correct_dir)
        for cls_name in cls_list:
            cls_path = os.path.join(correct_dir, cls_name)
            img_list = os.listdir(cls_path)
            for img_name in img_list:
                paths.append(os.path.join(cls_path, img_name))
                
        return paths
                
    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])
        bounding_box = pd.read_csv(os.path.join(self.root, 'bounding_boxes.txt'),
                                       sep=' ', names=['img_id', 'x', 'y', 'width', 'height'])
        
        
        data = images.merge(image_class_labels, on='img_id')
        data = data.merge(train_test_split, on='img_id')
        self.data = data.merge(bounding_box, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]
            
    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        bbox = [sample.x, sample.y, sample.width, sample.height]
        img = self.loader(path)
        
        w = img.width
        h = img.height
        
        if self.transform is not None:
            img = self.transform(img)
        _, rs_h, rs_w = img.shape
        
        bbox = self._get_resized_bbox(h, w, rs_h, rs_w, bbox)
    
        return img, target, bbox, path
    
    def get_segmentation_masks(self, paths):
        masks = []
        mask_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            transforms.ToTensor()
        ])
        seg_mask_base_path = os.path.join(self.root, 'segmentations')
        
        for path in paths:
            cls_name = path.split('/')[-2]
            img_name = path.split('/')[-1].split('.')[0]
            
            m = self.loader(os.path.join(seg_mask_base_path, '{}/{}.png'.format(cls_name, img_name)))
            m = mask_transform(m)
            masks.append(m)
        
        return torch.stack(masks, 0)
    
    def _get_resized_bbox(self, h, w, rs_h, rs_w, bbox):
        w_ratio, h_ratio = rs_w/w, rs_h/h
        bbox[0] = int(bbox[0]*w_ratio)
        bbox[1] = int(bbox[1]*h_ratio)
        bbox[2] = int(bbox[2]*w_ratio)
        bbox[3] = int(bbox[3]*h_ratio)
        
        return bbox

# class Cub2011(Dataset):
#     base_folder = '/archive/workspace/datasets/CUB_200_2011/images/'
#     url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
#     filename = 'CUB_200_2011.tgz'
#     tgz_md5 = '97eceeb196236b17998738112f37df78'

#     def __init__(self, root, train=True, transform=None, loader=default_loader, download=False):
#         self.root = os.path.expanduser(root)
#         self.transform = transform
#         self.loader = default_loader
#         self.train = train

#         if download:
#             self._download()

#         if not self._check_integrity():
#             raise RuntimeError('Dataset not found or corrupted.' +
#                                ' You can use download=True to download it')

#     def _load_metadata(self):
#         images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
#                              names=['img_id', 'filepath'])
#         image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
#                                          sep=' ', names=['img_id', 'target'])
#         train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
#                                        sep=' ', names=['img_id', 'is_training_img'])
#         bounding_box = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'bounding_boxes.txt'),
#                                        sep=' ', names=['img_id', 'x', 'y', 'width', 'height'])
        
#         data = images.merge(image_class_labels, on='img_id')
#         data = data.merge(train_test_split, on='img_id')
#         self.data = data.merge(bounding_box, on='img_id')

#         if self.train:
#             self.data = self.data[self.data.is_training_img == 1]
#         else:
#             self.data = self.data[self.data.is_training_img == 0]
        
#     def _check_integrity(self):
#         try:
#             self._load_metadata()
#         except Exception:
#             return False

#         for index, row in self.data.iterrows():
#             filepath = os.path.join(self.root, self.base_folder, row.filepath)
#             if not os.path.isfile(filepath):
#                 print(filepath)
#                 return False
#         return True

#     def _download(self):
#         import tarfile

#         if self._check_integrity():
#             print('Files already downloaded and verified')
#             return

#         download_url(self.url, self.root, self.filename, self.tgz_md5)

#         with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
#             tar.extractall(path=self.root)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         sample = self.data.iloc[idx]
#         path = os.path.join(self.root, self.base_folder, sample.filepath)
#         target = sample.target - 1  # Targets start at 1 by default, so shift to 0
#         bbox = [sample.x, sample.y, sample.width, sample.height]
#         img = self.loader(path)
        
#         w = img.width
#         h = img.height
        
#         if self.transform is not None:
#             img = self.transform(img)
#         _, rs_h, rs_w = img.shape
        
#         bbox = self._get_resized_bbox(h, w, rs_h, rs_w, bbox)
    
#         return img, target, bbox, path
    
#     def get_segmentation_masks(self, paths):
#         masks = []
#         mask_transform = torchvision.transforms.Compose([
#             torchvision.transforms.Resize(size=(224, 224)),
#             transforms.ToTensor()
#         ])
#         seg_mask_base_path = os.path.join(self.root, 'CUB_200_2011/segmentations')
        
#         for path in paths:
#             cls_name = path.split('/')[-2]
#             img_name = path.split('/')[-1].split('.')[0]
            
#             m = self.loader(os.path.join(seg_mask_base_path, '{}/{}.png'.format(cls_name, img_name)))
#             m = mask_transform(m)
#             masks.append(m)
        
#         return torch.stack(masks, 0)
    
#     def _get_resized_bbox(self, h, w, rs_h, rs_w, bbox):
#         w_ratio, h_ratio = rs_w/w, rs_h/h
#         bbox[0] = int(bbox[0]*w_ratio)
#         bbox[1] = int(bbox[1]*h_ratio)
#         bbox[2] = int(bbox[2]*w_ratio)
#         bbox[3] = int(bbox[3]*h_ratio)
        
#         return bbox