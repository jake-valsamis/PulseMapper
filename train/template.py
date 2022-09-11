import PIL
PIL.PILLOW_VERSION = PIL.__version__

import path
import torch
import torchvision
from functools import partial
import pandas as pd
import numpy as np
import random
import sys
sys.path.append('..\scripts')
sys.path.append('scripts')
from dataloader import *
from models import *
from opt import *
from learner import *
from data_augmentation import *
from metrics import *
from display import display_volume
from loss import *
from train import *

import os
import matplotlib
matplotlib.use('Agg')



"""
The trainer will use the defaults in learner.py unless they are overwritten here.
Change any defaults here by uncommenting the variable and changing the value
"""
"""Core parameters"""
defaults = dict(
                root_path      = Path(r'data'),
                img_folder     = 'Images MC',
                gt_folder      = 'Ground Truth',
                img_components = [f'fft{i}' for i in range(10)],
                gt_components  = ['rms'],
                extensions     = {'npy'},
                # img_tfms       = ToTensor(),
                bs             = 32, 
                lr             = 1e-3, 
                epochs         = 20, 
                
                
            
                ###------- From the Loss.ipynb/loss.py ---###
                #loss_func      = torch.nn.MSELoss()
                #loss_func        = BinaryWindowMSE(),
                 

                ###------- From the Model.ipynb/models.py """
                # model          = partial(DynamicResnet34Unet, base_model=torchvision.models.resnet34(pretrained=True))

                ###------- From Metrics.ipynb/metrics.py """
                #metrics        = [mean, mse, mae, pct_tolerance, abs_tolerance, auc_abs, auc_pct],

                ###------- From the Optimizer.ipynb/opt.py """
                # opt_func       = adam_opt(mom=0.9, mom_sqr=0.99, eps=1e-6, wd=1e-2)

                ###------- From the Data Augmentation.ipynb/data_augmentation.py """
                # tfms=[(rotate_batch, 1., {'degrees':(-20,20)}),
                #       (scale_batch, 0.2, {'x_mag':(0.8, 1.2)}),
                #       (skew_batch, 0.4,{'mag':(-0.05,0.05)}),
                #       (reflect_batch, 0.5, {'mirror_h':True}),
                #       (reflect_batch, 0.5, {'mirror_v':True}),
                #       (translate_batch, 0.3, {'x_mag':(-0.1,0.1), 'y_mag':(-0.1,0.1)}),
                #      ]

                ###------- From the Learner.ipynb/learner.py """
                # cbs = [ToDeviceCallback(torch.device('cuda')), 
                #        #NormalizeBatchCallback(norm_x=True),
                #       ]
                # cbfs = [ToFloatCallback, 
                #         Recorder, 
                #         StandardizeBatchCallback,
                #        ]
                
                
                ###----to define validation data ------
                 data_split_file = ['2092','2082','916', '914', '913', '2071','2067', '2102', '868', '837', '866','910', '843', '2081'],


                )

"""
update_parameters should contain a list of tuples.  Each tuple contains the name of the experiment and the dictionary of parameters to be changed
e.g. update_parameters = [('Test1', {'loss_func':WeightedMSELoss()}),
                          ('Test2', {'loss_func':nn.MSELoss()}),
                        ]
Test1 will use all the defaults, but replace loss_func with the weighted loss, whereas Test2 will use all the defaults, but replace the loss function
with MSELoss().

Note that you can use list comprehension to organize a series of experiments.  See below for examples
"""

#exp_name = f"fft{int( defaults['img_components'][0].split('fft')[-1]):02}To{int( defaults['img_components'][-1].split('fft')[-1]):02}_"

#update_parameters = [('fft00To29_MSEloss', {'loss_func':torch.nn.MSELoss()}),
#                      ('fft00To29_wMSE2to1', {'loss_func':BinaryWindowMSE_pos2_neg1()}),
#                      ('fft00To29_wMSEp8top2', {'loss_func':BinaryWindowMSE()}),
#                         ]
update_parameters = [('fft00To9_wMSE2to1', {'loss_func':BinaryWindowMSE_pos2_neg1()})]
train_models(update_parameters, defaults, exp_name = exp_name, output_images=True)

"""
-------------------------------
Specific examples
-------------------------------
"""

"""To train models with successively more input channels"""
# channels = ['mean','std','iqr','FAnoise0','fft5', 'fft6','fft7','fft8',
#             'FAcomp1', 'FAcomp2', 'FAcomp3', 'FAcomp4', 'FAcomp5']
# update_parameters=[(f'Weigh{i:03}-{channels[0]}-{channels[i]}',{'img_components':channels[:i+1]}) for i in range(len(channels))]
# train_models(update_parameters, exp_name='Multi-Output')


"""Adding a new metric to the default list"""
# def new_metric(pred, target): return ((pred>target)*1).mean()
# metrics.append(new_metric)
# update_parameters[('Defaults', {}]
# train_models(update_parameters, exp_name = 'NewMetric')


"""More complex experiments - Goal is to combine adding more input channels with changing the model"""
#Compare Models
# channels = ['mean','std','iqr','FAnoise0','fft5', 'fft6','fft7','fft8',
#             'FAcomp1', 'FAcomp2', 'FAcomp3', 'FAcomp4', 'FAcomp5']

# update_parameters=[(f'Unet - {i:03}-{channels[0]}-{channels[i]}',{'img_components':channels[:i+1],
#                                                           }) for i in range(len(channels))]

# update_parameters+=[(f'Simple1x1 - {i:03}-{channels[0]}-{channels[i]}',{'img_components':channels[:i+1],
#                                                             'model':Simple1x1,
#                                                           }) for i in range(len(channels))]

# update_parameters+=[(f'SimpleResnet - {i:03}-{channels[0]}-{channels[i]}',{'img_components':channels[:i+1],
#                                                             'model':SimpleResnet,
#                                                           }) for i in range(len(channels))]

# train_models(update_parameters, exp_name='Multi-Output')




