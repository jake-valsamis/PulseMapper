# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 10:20:48 2021

@author: nhaq
"""
import PIL
PIL.PILLOW_VERSION = PIL.__version__

import path
import torch
# import torchvision
# from functools import partial
import pandas as pd
import numpy as np
import random
import sys
sys.path.append('..\scripts')
sys.path.append('scripts')
from dataloader import *

from data_augmentation import *


import os
import pickle

cv_fold =5 ## 5-fold cross-validation

output_folder = 'data/CVfolds/'

root_path      = Path(r'data')
img_folder     = 'Images MC'
img_components = [f'fft{i}' for i in range(10)]
gt_components  = ['rms']
extensions     = {'npy'}
img_tfms       = ToTensor()

il = MRIImageList.from_files(root_path/img_folder, img_components, extensions, img_tfms=img_tfms)
users = list({get_user(fp) for fp in il.items})
random.shuffle(users)

valid_size = int(np.ceil(len(users)//cv_fold))

if not os.path.exists(output_folder):  os.mkdir(output_folder)

for cv in range(cv_fold):
    valid_users = users[cv*valid_size:min((cv+1)*valid_size, len(users))]
    print(valid_users)
    sd = SplitData.split_by_user_fold( il, valid_users)
    pickle.dump(sd, open(f"{output_folder}/cv{cv}.pkl", 'wb'))
    
    
