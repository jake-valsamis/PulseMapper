
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/Data Augmentation.ipynb
import numpy as np
import torch
import math
import torch.nn.functional as F
from pathlib import Path

"""
Perform a transformation on a batch using affine trandformations.

Essentially, you create a grid (using F.affine_grid) that maps all of the new coordinates from your original image
to the new tensor.  In theory, you could make your own function using linspace but you're guaranteed to preserve
certain features (such as the fact that lines remain parallel) if you stick to affine transformations

E.g. If your original image is 200x200 and your affine grid has 0.3172, 0.312 at pixel (0,0),
it will look for the 4 nearest points around the (50+30/2)% mark of the original image (130.0614, 130.0614).
The four closest pixels are (130,130), (131, 130), (130, 131), (131, 131), so you will determine the new
value based on these four pixels and your preferred mode.
"""

#Matrix Definitions
def rotation_matrix(thetas):
    """
    Creates a tensor of the form:
    [cos(theta)  sin(theta)  0
     sin(theta)  cos(theta)  0]
    """
    thetas.mul_(math.pi/180)
    rows = [torch.stack([thetas.cos(),             thetas.sin(),             torch.zeros_like(thetas)], dim=1),
            torch.stack([-thetas.sin(),            thetas.cos(),             torch.zeros_like(thetas)], dim=1),]
    return torch.stack(rows, dim=1)

def skew_matrix(mags):
    """
    Creates a tensor of the form:
    [1           mags        0
     mags        1           0]
    """
    rows = [torch.stack([torch.ones_like(mags),             mags,             torch.zeros_like(mags)], dim=1),
            torch.stack([mags,            torch.ones_like(mags),             torch.zeros_like(mags)], dim=1),]
    return torch.stack(rows, dim=1)

def translation_matrix(x_mags, y_mags):
    """
    Creates a tensor of the form:
    [1           0           x_mag
     0           1           y_mag]
    """
    rows = [torch.stack([torch.ones_like(x_mags),  torch.zeros_like(x_mags),  x_mags], dim=1),
            torch.stack([torch.zeros_like(x_mags), torch.ones_like(x_mags),   y_mags], dim=1),]
    return torch.stack(rows, dim=1)
def scale_matrix(x_mags, y_mags):
    """
    Creates a tensor of the form:
    [x_scale     0           0
     0           y_scale     0]
    """
    rows = [torch.stack([x_mags,                   torch.zeros_like(x_mags),  torch.zeros_like(x_mags)], dim=1),
            torch.stack([torch.zeros_like(x_mags), y_mags,                    torch.zeros_like(x_mags)], dim=1),]
    return torch.stack(rows, dim=1)


#Matrix intermediate functions
def rotate_uniform(base, degrees): return rotation_matrix(base.fill_(degrees))
def rotate_random(base, degrees): return rotation_matrix(base.uniform_(-degrees, degrees))
def translate_uniform(base, x_mag, y_mag): return translation_matrix(base.clone().fill_(x_mag), base.fill_(y_mag))
def skew_uniform(base, mag): return skew_matrix(base.fill_(mag))
def skew_uniform(base, mag): return skew_matrix(base.fill_(mag))
def scale_uniform(base, x_scale, y_scale): return scale_matrix(base.clone().fill_(x_scale), base.fill_(y_scale))
def custom_transform(base, matrix): return torch.stack([torch.tensor(matrix)]*base.shape[0]).cuda()

#generalized affine transformation
def affine_tsfm(x, size, matrix_func, mode = 'nearest', padding_mode = 'zeros', to_cuda=True, *args, **kwargs):
    if not isinstance(x, torch.Tensor): x = torch.tensor(x)
    size = (size,size) if isinstance(size, int) else tuple(size)
    size = (x.size(0),x.size(1)) + size
    m = matrix_func(x.new(x.size(0)), *args, **kwargs)
    grid = F.affine_grid(m, size, align_corners=False)
    if to_cuda: x, grid = x.cuda(), grid.cuda()
    return F.grid_sample(x, grid, align_corners = False, mode = mode, padding_mode = 'zeros')

#front-end functions
def rotate_batch(x, size, degrees, random = False):
    matrix_func = rotate_random if random else rotate_uniform
    return affine_tsfm(x, size, matrix_func, degrees = degrees)
def skew_batch(x, size, mag): return affine_tsfm(x, size, skew_uniform, mag=mag)
def translate_batch(x, size, x_mag, y_mag): return affine_tsfm(x, size, translate_uniform, x_mag=x_mag, y_mag=y_mag)
def scale_batch(x, size, x_mag, y_mag=None):
    y_mag = x_mag if y_mag is None else y_mag
    return affine_tsfm(x, size, scale_uniform, x_scale=x_mag, y_scale=y_mag)
def reflect_batch(x, size, mirror_h = False, mirror_v = False):
    mirror_h = -1 if mirror_h else 1
    mirror_v = -1 if mirror_v else 1
    return affine_tsfm(x, size, scale_uniform, x_scale=mirror_h, y_scale=mirror_v)
def custom_affine_batch(x, size, matrix): return affine_tsfm(x, size, custom_transform, matrix=matrix)
def resize_batch_cpu(x, size): return affine_tsfm(x,size,scale_uniform,'bilinear',to_cuda=False, x_scale = 1, y_scale = 1)

def load_grid(x_name, y_name, root_path = Path('../data/Custom Masks/')):
    xx = imageio.imread(root_path/f'{x_name}.png')
    yy = imageio.imread(root_path/f'{y_name}.png').T
    return np.repeat(np.dstack([xx, yy])[None, ...], 68, axis = 0).astype(np.double)/32768 - 1

def tfms_from_grid(x, grid, to_gpu=True, mode='nearest'):
        if not isinstance(x, torch.Tensor):       x = torch.tensor(x)
        if not isinstance(grid, torch.Tensor):    grid = torch.tensor(grid)
        if to_gpu:  x, grid = x.cuda(), grid.cuda()
        return F.grid_sample(x, grid, align_corners=False, mode=mode, padding_mode='zeros')