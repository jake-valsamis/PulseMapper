# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 17:25:30 2021

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
# with path.Path(".."):
#     from scripts.display import display_volume
    
import nibabel as nib
import pandas as pd
import glob

import pywt

#from sklearn import decomposition
#from sklearn.cluster import MiniBatchKMeans





def get_mean(arr, axis=0): return arr.mean(axis=axis)
def get_std(arr, axis=0): return arr.std(axis=axis)
def get_pctile(arr, pct, axis=0): return np.percentile(arr, pct, axis=axis)
def get_iqr(arr, axis=0): return get_pctile(arr, 75) - get_pctile(arr, 25)
def get_fft(arr, nbins=100):
    res = np.apply_along_axis(partial(np.fft.fft, n=nbins), axis=0, arr=arr.reshape((arr.shape[0],-1)))
    return np.real(res.reshape(nbins, *arr.shape[1:]))[:nbins,...]
def fft_slice(arr, n=3): return np.real(get_fft(arr)[n])
def get_comp_noise(arr, ep, n=6): return get_component_analysis(arr, ep, n, True)[1]
def get_comp(arr, ep, n=6): return get_component_analysis(arr, ep, n, True)[0]



##def dwt_slice(arr, level=3): return pywt.downcoef(part, data, wavelet, mode='symmetric', level=1)




def get_component_analysis(img, estimator_params, n_components=6, return_noise=True):
    h,w = img.shape[-2:]
    vol_shape = img.shape[1:]
    t=img.shape[0]
    img_slice = img[:,...].reshape(t,-1)
    n_samples, n_features = img_slice.shape
    img_centered = img_slice - img_slice.mean(axis=0)
    img_centered-=img_centered.mean(axis=1).reshape(n_samples, -1)
    name, estimator,center = estimator_params
    estimator.fit(img_centered if center else img_slice)
    components_=estimator.cluster_centers_ if hasattr(estimator, 'cluster_centers_') else estimator.components_
    components = components_.reshape(n_components, *vol_shape)
    if (hasattr(estimator, 'noise_variance_') and estimator.noise_variance_.ndim>0):
        noise_variance = estimator.noise_variance_
        noise_variance=noise_variance.reshape(1, *vol_shape)
    else: noise_variance = None
        
    return components, noise_variance



proc_funcs = {'mean':  get_mean,
              'std':   get_std,
              'iqr':   get_iqr,
              'fft0':  partial(fft_slice, n=0),
              'fft1':  partial(fft_slice, n=1),
              'fft2':  partial(fft_slice, n=2),
              'fft3':  partial(fft_slice, n=3),
              'fft4':  partial(fft_slice, n=4),
              'fft5':  partial(fft_slice, n=5),
              'fft6':  partial(fft_slice, n=6),
              'fft7':  partial(fft_slice, n=7),
              'fft8':  partial(fft_slice, n=8),
              'fft9':  partial(fft_slice, n=9),
              'fft10':  partial(fft_slice, n=10),
              'fft11':  partial(fft_slice, n=11),
              'fft12':  partial(fft_slice, n=12),
              'fft13':  partial(fft_slice, n=13),
              'fft14':  partial(fft_slice, n=14),
              'fft15':  partial(fft_slice, n=15),
              'fft16':  partial(fft_slice, n=16),
              'fft17':  partial(fft_slice, n=17),
              'fft18':  partial(fft_slice, n=18),
              'fft19':  partial(fft_slice, n=19),
              'fft20':  partial(fft_slice, n=20),
              'fft21':  partial(fft_slice, n=21),
              'fft22':  partial(fft_slice, n=22),
              'fft23':  partial(fft_slice, n=23),
              'fft24':  partial(fft_slice, n=24),              
              'fft25':  partial(fft_slice, n=25),                       
              'fft26':  partial(fft_slice, n=26),                      
              'fft27':  partial(fft_slice, n=27),                      
              'fft28':  partial(fft_slice, n=28),                                    
              'fft29':  partial(fft_slice, n=29),                                                  
              'fft30':  partial(fft_slice, n=30),
              'fft31':  partial(fft_slice, n=31),
              'fft32':  partial(fft_slice, n=32),
              'fft33':  partial(fft_slice, n=33),
              'fft34':  partial(fft_slice, n=34),
              'fft35':  partial(fft_slice, n=35),
              'fft36':  partial(fft_slice, n=36),
              'fft37':  partial(fft_slice, n=37),
              'fft38':  partial(fft_slice, n=38),
              'fft39':  partial(fft_slice, n=39),
              'fft40':  partial(fft_slice, n=40),
              'fft41':  partial(fft_slice, n=41),
              'fft42':  partial(fft_slice, n=42),
              'fft43':  partial(fft_slice, n=43),
              'fft44':  partial(fft_slice, n=44),
              'fft45':  partial(fft_slice, n=45),
              'fft46':  partial(fft_slice, n=46),
              'fft47':  partial(fft_slice, n=47),
              'fft48':  partial(fft_slice, n=48),
              'fft49':  partial(fft_slice, n=49),                                                  
              'fft50':  partial(fft_slice, n=50),
              'fft51':  partial(fft_slice, n=51),
              'fft52':  partial(fft_slice, n=52),
              'fft53':  partial(fft_slice, n=53),
              'fft54':  partial(fft_slice, n=54),
              'fft55':  partial(fft_slice, n=55),
              'fft56':  partial(fft_slice, n=56),
              'fft57':  partial(fft_slice, n=57),
              'fft58':  partial(fft_slice, n=58),
              'fft59':  partial(fft_slice, n=59),
              'fft60':  partial(fft_slice, n=60),
              'fft61':  partial(fft_slice, n=61),
              'fft62':  partial(fft_slice, n=62),
              'fft63':  partial(fft_slice, n=63),
              'fft64':  partial(fft_slice, n=64),
              'fft65':  partial(fft_slice, n=65),
              'fft66':  partial(fft_slice, n=66),
              'fft67':  partial(fft_slice, n=67),
              'fft68':  partial(fft_slice, n=68),
              'fft69':  partial(fft_slice, n=69),
              'fft':   get_fft, 
              #'FAnoise':  partial(get_comp_noise, ep=estimator_params['FA']),
              #'FAcomp':   partial(get_comp, ep=estimator_params['FA'])
             }




def create_list_of_paths(root_path, folder, filename_pattern, list_of_paths=None):
    wdir=os.getcwd()
    if not list_of_paths:
        list_of_paths = []
    os.chdir(f"{root_path.as_posix()}/{folder}")
    for name in glob.glob(filename_pattern):
        list_of_paths.append(Path(os.path.abspath(name)))
    os.chdir(wdir)
    return list_of_paths



# def get_all_files(path):
#     return list(os.walk(path))

# def ls(self): return list(self.iterdir())
# Path.ls = ls


def open_mri(filepath):
    img = nib.load(filepath).get_fdata()
    transpose_array = (3,2,0,1) if img.ndim == 4 else (2,0,1) 
    img = np.transpose(img, transpose_array)[...,None,:,:]
    if img.shape[0] == 1: img = np.squeeze(img, 0)
    return img




# def output_slices(root_path, img_stack, gts_stack, img_root, gt_root, output_folder, proc_func = None):
#     """ Outputs the individual image slices (for now) for each stack into the appropriate folders
#     Expects an image array of shape:  T x N x C x W x H 
#     Expects a  gt    array of shape:      N x 1 x W x H

#     Image outputs will be of shape:           C x W x H, where C is the number of procs
#     GT outputs will be of shape:              1 x W x H
#     """
#     if proc_func is not None: img_stack = proc_func(img_stack)   
#     img_save_path = root_path/img_root/output_folder;    img_save_path.mkdir(exist_ok=True,parents=True) 
#     gts_save_path = root_path/gt_root /output_folder;    gts_save_path.mkdir(exist_ok=True,parents=True) 

#     for slice_num in range(img_stack.shape[0]):
#         np.save(img_save_path/f'{slice_num}', img_stack[slice_num,...])
# #         np.save(gts_save_path/f'{slice_num}', gts_stack[slice_num,0,...]) #Maybe include axis 1??
#         np.save(gts_save_path/f'{slice_num}', gts_stack[slice_num,...])
#     return True



def output_multiecho_slices_wGT(img_stack, gts_stack, img_root, gt_root, user_id, echo_id, proc_func = None):
    """ Outputs the individual image slices (for now) for each stack into the appropriate folders
    Expects an image array of shape:  T x N x C x W x H 
    Expects a  gt    array of shape:      N x 1 x W x H

    Image outputs will be of shape:           C x W x H, where C is the number of procs
    GT outputs will be of shape:              1 x W x H
    """
    if proc_func is not None: img_stack = proc_func(img_stack)   
    img_save_path = img_root/user_id;    img_save_path.mkdir(exist_ok=True,parents=True) 
    gts_save_path = gt_root /user_id;    gts_save_path.mkdir(exist_ok=True,parents=True) 

    for slice_num in range(img_stack.shape[0]):
        np.save(img_save_path/f'{echo_id}_{slice_num}', img_stack[slice_num,...])
#         np.save(gts_save_path/f'{slice_num}', gts_stack[slice_num,0,...]) #Maybe include axis 1??
        np.save(gts_save_path/f'{echo_id}_{slice_num}', gts_stack[slice_num,...])
    return True


def output_multiecho_slices(img_stack, img_root_folder,  user_id, echo_id, proc_func = None):
    """ Outputs the individual image slices (for now) for each stack into the appropriate folders
    Expects an image array of shape:  N x C x W x H 
    Image outputs will be of shape:       C x W x H
    For most applications, C will be 1 
    """
    if proc_func is not None: img_stack = proc_func(img_stack)   
    img_save_path = img_root_folder/user_id;    img_save_path.mkdir(exist_ok=True,parents=True) 
    #print(img_save_path, img_stack.shape)
    for slice_num in range(img_stack.shape[0]):
        np.save(img_save_path/f'{echo_id}_{slice_num}', img_stack[slice_num,...])
        
    #####print(img_root_folder, user_id, echo_id, img_save_path)
    return True



def process_multiecho_slices(img_stack, img_root_folder, output_folder, echo_id, proc_func = None):
    if img_stack.ndim==4: output_multiecho_slices(img_stack, img_root_folder, output_folder, echo_id, proc_func = None)
    elif img_stack.ndim==5:
        for i in range(img_stack.shape[0]):
            new_output_folder = output_folder.parent/f"{output_folder.name}{i}"
            output_multiecho_slices(img_stack[i,...], img_root_folder, new_output_folder, echo_id, proc_func = None)


def process_all(raw_img, procs = ['mean', 'std', 'iqr']):
    """
    Expects an image array of shape:  T x N x 1 x W x H 
    Proc   outputs will be of shape:      N x 1 x W x H (or more if a multi-stack) 
    Image  outputs will be of shape:      N x C x W x H
    """
    img_stack = [proc_funcs[name](raw_img) for name in procs]
    return np.concatenate(img_stack, axis=1)




class multi_echo_data_generator():
    def __init__(self, img_info, gt_info, img_file_pattern, gt_pattern):
        self.img_info = img_info
        self.gt_info = gt_info
        self.img_file_pattern = img_file_pattern
        self.gt_pattern = gt_pattern
        
    def create_multiecho_training_data(self, root_path, img_src, gt_src, procs, echo_index=[1], individual_files = True, process_gt=True):
        """ Outputs the individual image slices (for now) for each stack into the appropriate folders
        Expects an image array of shape:  T x N x C x W x H 
        Expects a  gt    array of shape:      N x 1 x W x H
    
        Image outputs will be of shape:           C x W x H, where C is the number of procs
        GT outputs will be of shape:              1 x W x H
        """
        
        # img_folder, img_name_fnc, img_output_folder = img_info[img_src]
        # gt_folder, gt_name_fnc, gt_output_folder = gt_info[gt_src]
        
        # raw_img_path = root_path/img_folder
        # raw_gt_path = root_path/gt_folder
    
        # raw_imgs, raw_gts = list(raw_img_path.iterdir()), list(raw_gt_path.iterdir())
    
        img_folder_fnc, img_name_fnc, echo_name_fnc, img_output_folder = self.img_info[img_src]
        gt_folder_fnc, gt_name_fnc, gt_output_folder = self.gt_info[gt_src]
    
        raw_imgs = []
        raw_gts = []
        
        for echo_id in echo_index:
            print(f"---------- Echo : {echo_id} ------------")
            raw_imgs = create_list_of_paths(root_path, img_folder_fnc(echo_id), self.img_file_pattern, list_of_paths=raw_imgs)
            raw_gts = create_list_of_paths(root_path, gt_folder_fnc(echo_id), self.gt_pattern, list_of_paths=raw_gts)
        
        for raw_img_path in raw_imgs:
            user_id = img_name_fnc(raw_img_path)
            echo_id = echo_name_fnc(raw_img_path)
            print(f"Echo {echo_id}: {user_id}")
            
            gt_folder = gt_folder_fnc(echo_id)
            raw_gt_path = root_path/gt_folder
            
            raw_img = open_mri(raw_img_path)
            gt_path = raw_gt_path/gt_name_fnc(echo_id, user_id)
            if gt_path.exists():
                if process_gt:
                    gt = open_mri(gt_path)
                if individual_files:
                    for proc_name in procs:
                        processed_image = proc_funcs[proc_name](raw_img)
                        subfolder = Path(user_id)/proc_name
                        process_multiecho_slices(processed_image, img_output_folder, subfolder, echo_id)
                    if process_gt:
                        gt_subfolder = Path(user_id)/gt_src.replace('gt_','')
                        process_multiecho_slices(gt, gt_output_folder, gt_subfolder, echo_id)
    
                else:
                    processed_all = process_all(raw_img, procs)
                    output_multiecho_slices_wGT(processed_all, gt, img_output_folder, gt_output_folder, user_id, echo_id)
            else:
                print (f"Missing Echo {echo_id}: {user_id}")
