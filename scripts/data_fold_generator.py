# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:08:40 2021

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

class data_fold_generator():
     def __init__(self, defaults):
         self.root_path      = defaults['root_path'] 
         self.img_folder     = defaults['img_folder'] 
         self.img_components = defaults['img_components'] 
         self.extensions     = defaults['extensions']
         self.img_tfms       = ToTensor()
         
     def generate_folds(self, cv_fold=5, output_folder = 'data/output/CVfolds/'):
         il = MRIImageList.from_files(self.root_path/self.img_folder, self.img_components, self.extensions, img_tfms=self.img_tfms)
         users = list({get_user(fp) for fp in il.items}) 
         random.shuffle(users) 
         valid_size = int(np.ceil(len(users)//cv_fold)) 
         
         if not os.path.exists(output_folder):  os.mkdir(output_folder) 
         
         for cv in range(cv_fold):
             valid_users = users[cv*valid_size:min((cv+1)*valid_size, len(users))]
             print(valid_users)
             sd = SplitData.split_by_user_fold( il, valid_users)
             pickle.dump(sd, open(f"{output_folder}/cv{cv}.pkl", 'wb'))
             
     def generate_one_split(self, output_folder = 'data/output/OneSplit/', valid_users=['2092','2067','834','2060', '917','2102','2037','2078','2068','2094','866']):
         il = MRIImageList.from_files(self.root_path/self.img_folder, self.img_components, self.extensions, img_tfms=self.img_tfms)
         
         sd = SplitData.split_by_user_fold( il, valid_users) 
         pickle.dump(sd, open(f"{output_folder}/split.pkl", 'wb'))
             