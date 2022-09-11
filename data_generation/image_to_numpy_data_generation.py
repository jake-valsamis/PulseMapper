# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:49:51 2021

@author: nhaq
"""


import numpy as np
import imageio
import path
from bokeh.io import output_notebook
from functools import partial
import matplotlib.pyplot as plt
from pathlib import Path
import os

    
import nibabel as nib
import pandas as pd
import glob

from datagen_utils import *


    



img_mc_pattern=f"echo[0-9]_retroicor_mc_bet_RISE*[?0-9].nii*"
gt_pattern=f"echo[0-9]_100vol_cardiac_RISE*[?0-9]_RMS.nii*"

output_folder = '../data/'
output_folder = os.path.abspath(output_folder)
output_folder = Path(output_folder)




img_info = {'img_mc': (lambda i: f"Echo{i}_respretroicor_mc_bet_20170810", 
                     lambda x: x.name.split('_')[4][4:].replace('.nii.gz', ""),
                     lambda x: x.name.split('_')[0][4:].replace('echo',""),
                     output_folder/'Images MC'),
             }

gt_info = {'gt_rms': (lambda i:f"Echo{i}_Cardiac_Maps_PE", 
                      lambda i, x: f"echo{i}_100vol_cardiac_RISE{x}_RMS.nii.gz",
                     output_folder/'Ground Truth'), 
             }




gt_src = 'gt_rms'
img_src = 'img_mc'

"""Make sure to set your root path!!"""
root_path = '../src/'
root_path = os.path.abspath(root_path)
root_path = Path(root_path)

procs=[f'fft{i}' for i in range(0,10)]

echo_index =[1,2,3]


datagen = multi_echo_data_generator(img_info, gt_info, img_file_pattern=img_mc_pattern, gt_pattern=gt_pattern)


#datagen.create_multiecho_training_data(root_path, img_src, gt_src, procs, echo_index)

datagen.create_multiecho_training_data(root_path, img_src, gt_src, procs, echo_index, process_gt=True)
#change process_gt from False to True to make data\Ground Truth folder 


### check and then flip everything
_ , _ , _, datafolder = img_info[img_src]


for sbdir in datafolder.iterdir():
    for procdir in procs:
        fullpath = sbdir/procdir
        for filepath in fullpath.iterdir():
            data = np.load(filepath)
            np.save( filepath,   np.fliplr(data))
    


