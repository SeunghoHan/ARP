LAYER_INFO_KEY = 'layer'
SHAPE_INFO_KEY = 'shape'
LAYER_NAME_KEY = 'name'

NAME_OF_EXPLAINER_LRP = 'LRP'
NAME_OF_EXPLAINER_CLRP = 'CLRP'
NAME_OF_EXPLAINER_SGLRP = 'SGLRP'
NAME_OF_EXPLAINER_SLRP = 'SLRP'

LAYER_SEP_TYPE_FULL = 'FULL'
LAYER_SEP_TYPE_LAST = 'LAST'
LAYER_SEP_TYPE_CORE = 'CORE'


sz_batch_train = 80
# sz_batch_train = 40
sz_batch_val = 20
max_stopping_cnt = 3


"""
Hyper-parameters form CLAM and CLA-layer
 -> nb_attrs, sz_pathces, sz_attrs, lr, max_epoches, strength, alpha, beta
     - AUTO: set values in run time 
     - MANUAL: 
     - full: use all conv layers in target model
"""

nb_attrs = 'MANUAL' # 'AUTO'
nb_attrs_vgg16_full = [15, 15, 10, 10, 8, 8, 8, 5, 5, 5, 5, 5, 5]
nb_attrs_vgg16_core = [15, 10, 8, 5, 5]
# 6, 8, 12, 6
nb_attrs_resnet50 = [13,11,13,11,13,11, 9,7,9,7,9,7,9,7, 7,5,7,5,7,5,7,5,7,5,7,5, 7,5,7,5,7,5]
# 3, 4, 6, 3
nb_attrs_resnet50_core = [11,11,11, 7,7,7,7, 5,5,5,5,5,5, 5,5,5]

sz_patches = 'MANUAL' # 'AUTO'
sz_patches_vgg16_cub_full = [14, 14, 11, 11, 7, 7, 7, 5, 5, 5, 3, 3, 3]
sz_patches_vgg16_imagenet_full = [14, 14, 11, 11, 7, 7, 7, 5, 5, 5, 4, 4, 4]
sz_patches_vgg16_imagenet_core = [14, 11, 7, 5, 4]
sz_patches_resnet50_cub = [7,7,7,7,7,7, 5,5,5,5,5,5,5,5, 3,3,3,3,3,3,3,3,3,3,3,3, 3,3,3,3,3,3]
sz_patches_resnet50_imagenet_core = [7,7,7, 5,5,5,5, 4,4,4,4,4,4, 4,4,4]

sz_attrs = 'MANUAL' # 'AUTO'
sz_attrs_vgg16_cub_full = [6, 6, 5, 5, 3, 3, 3, 2, 2, 2, 1, 1, 1]
sz_attrs_vgg16_imagenet_full = [6, 6, 5, 5, 3, 3, 3, 2, 2, 2, 1, 1, 1]
sz_attrs_vgg16_imagenet_core = [6, 5, 3, 2, 1]
sz_attrs_resnet50_cub = [3,3,3,3,3,3, 2,2,2,2,2,2,2,2, 1,1,1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1]
sz_attrs_resnet50_imagenet_core = [3,3,3,3,3,3, 2,2,2,2, 1,1,1,1, 1,1,1,1]

strength = 'MANUAL' # 'AUTO'
# strength_vgg16_full = [1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.18, 1.18, 1.18, 1.2, 1.3, 1.4]
strength_vgg16_full = [1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.15, 1.18, 1.18, 1.18, 1.2, 1.25, 1.3]
strength_vgg16_core = [1.1, 1.15,1.15, 1.18, 1.25]
strength_resnet50 = [1.1,1.1,1.1,1.1,1.1,1.1, 1.1,1.1,1.1,1.1,1.1,1.1,1.1,1.1, 1.15,1.15,1.15,1.15,1.15,1.15,1.15,1.15,1.15,1.15,1.15,1.15, 1.20,1.2,1.2,1.2,1.2,1.2]
strength_resnet50_core = [1.1,1.1,1.1, 1.1,1.1,1.1,1.1, 1.15,1.15,1.15,1.15,1.15,1.15, 1.20,1.2,1.2]

alpha = 'MANUAL' # 'AUTO'
alpha_vgg16_full = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
alpha_vgg16_core = [0.1, 0.1, 0.1, 0.1, 0.1]
alpha_resnet50 = [0.1,0.1,0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1,0.1]
alpha_resnet50_core = [0.1,0.1,0.1, 0.1,0.1,0.1,0.1, 0.1,0.1,0.1,0.1,0.1,0.1, 0.1,0.1,0.1]

beta = 'MANUAL' # 'AUTO'
beta_vgg16_full = [0.5, 0.5, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]
beta_vgg16_core = [0.5, 0.6, 0.6, 0.6, 0.6]
beta_resnet50 = [0.6,0.6,0.6,0.6,0.6,0.6, 0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6, 0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6,0.6, 0.6,0.6,0.6,0.6,0.6,0.6]
beta_resnet50_core = [0.6,0.6,0.6, 0.6,0.6,0.6,0.6, 0.6,0.6,0.6,0.6,0.6,0.6, 0.6,0.6,0.6]


lr = 'AUTO'
lr_vgg16_full = []
lr_vgg16_last_conv = []
# {0: 0.015, 2: 0.015, 5: 0.025, 7: 0.025, 10: 0.015, 12: 0.015, 14: 0.015, 17: 0.01, 19: 0.01, 21: 0.01, 24: 0.005, 26: 0.005, 28: 0.005}

max_epoches = 100


