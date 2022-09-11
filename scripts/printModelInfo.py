
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/Model.ipynb
"""Note: there is an issue with torchvision's import of Pillow.  If you have an older version of torchvision
or a new version of pillow (>7), and cannot change either, you need to include the fix below"""
import PIL
PIL.PILLOW_VERSION = PIL.__version__

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
#from fastai.layers import PixelShuffle_ICNR, conv_layer
from functools import partial



def conv_layer(ni:int, nf:int, ks:int=3, stride:int=1, padding:int=None, bias:bool=None,
               bn=True,  norm_type = None, use_activ=True, leaky=None,
               init=nn.init.kaiming_normal_):
    "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
    if padding is None: padding = (ks-1)//2
    bn = True
    if bias is None: bias = not bn
    conv_func = nn.Conv2d
    conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
    if   norm_type is not None:   conv = nn.utils.weight_norm(conv)
    layers = [conv]
    relu = nn.LeakyReLU(leaky) if leaky is not None else nn.ReLU()
    if use_activ: layers.append(relu)
    if bn: layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)
def init_default(m, func=nn.init.kaiming_normal_):
    "Initialize `m` weights with `func` and set `bias` to 0."
    if func and hasattr(m, 'weight'): func(m.weight)
    with torch.no_grad():
        if getattr(m, 'bias', None) is not None: m.bias.fill_(0.)
    return m

def icnr_init(x, scale=2, init=nn.init.kaiming_normal_):
    "ICNR init of `x`, with `scale` and `init` function"
    ni,nf,h,w = x.shape
    ni2 = int(ni/(scale**2))
    k = init(x.new_zeros([ni2,nf,h,w])).transpose(0, 1)
    k = k.contiguous().view(ni2, nf, -1)
    k = k.repeat(1, 1, scale**2)
    return k.contiguous().view([nf,ni,h,w]).transpose(0, 1)

class PixelShuffle_ICNR(nn.Sequential):
    "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`."
    def __init__(self, ni, nf=None, scale=2, blur=False, leaky=None, norm_type=True):
        super().__init__()
        nf = ni if nf is None else nf
        layers = [conv_layer(ni, nf*(scale**2), ks=1, norm_type=norm_type),
                  nn.PixelShuffle(scale)]
        layers[0][0].weight.data.copy_(icnr_init(layers[0][0].weight.data))
        if blur: layers += [nn.ReplicationPad2d((1,0,1,0)), nn.AvgPool2d(2, stride=1)]
        super().__init__(*layers)



def conv2d1x1(c_in, c_out, kernel_size=(1,1), stride = (1,1), padding = 0):
    return nn.Conv2d(c_in, c_out, kernel_size, stride, padding)


class Simple1x1(nn.Module):
    def __init__(self, c_in, c_out, nh = None):
        super().__init__()
        self.nh = [20,20,10] if nh is None else nh
        self.c_in, self.c_out = c_in, c_out
        self.sizes = [self.c_in] + self.nh + [self.c_out]
        layers = []
        for channels_in, channels_out in zip(self.sizes[:-1], self.sizes[1:]):
            layers.append(conv2d1x1(channels_in, channels_out))
            layers.append(nn.BatchNorm2d(channels_out))
            layers.append(nn.LeakyReLU())
        layers = layers[:-1] #remove the last relu
        self.model = nn.Sequential(*layers)
    def forward(self, x): return self.model(x)
from torchsummary import summary 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Simple1x1(9,1).to(device)
summary(model, (9,80,80))

def conv2d(c_in, c_out, kernel_size=(3,3), stride = (1,1), padding = 1):
    return nn.Conv2d(c_in, c_out, kernel_size, stride, padding)


class ResNetBlock(nn.Module):
    def __init__(self, c_in, c_out=None):
        super().__init__()
        if c_out is None: c_out = c_in
        self.c_in, self.c_out = c_in, c_out
        self.conv1 = conv2d(c_in, c_out)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.relu = nn.LeakyReLU()
        self.conv2 = conv2d(c_out, c_out)
        self.bn2 = nn.BatchNorm2d(c_out)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)-0.5   #Helps to keep mean of 0 since the mean will be ~0.5 after relu
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)-0.5
        out+= identity
        out = self.relu(out)
        return out



class SimpleResnet(nn.Module):
    def __init__(self, c_in, c_out, nh = None, ):
        super().__init__()
        self.nh = [10,10] if nh is None else nh
        self.c_in, self.c_out = c_in, c_out
        sizes = self.nh + [self.c_out]

        self.conv1 = conv2d(c_in, self.nh[0])
        self.bn1 = nn.BatchNorm2d(self.nh[0])
        self.relu = nn.LeakyReLU()
        self.conv2 = conv2d(self.nh[-1], self.c_out)
        self.bn2 = nn.BatchNorm2d(self.c_out)
        self.blocks = []
        for ch in self.nh:
            self.blocks.append(ResNetBlock(ch, ch))
        self.model = nn.Sequential(*self.blocks)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)-0.5   #Helps to keep mean of 0 since the mean will be ~0.5 after relu
        out = self.relu(out)

        out = self.model(out)

        out = self.conv2(out)
        out = self.bn2(out)-0.5
        out = self.relu(out)

        return out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleResnet(49,1).to(device)
summary(model, (49,80,80))

def cut_model(model, beginning=0, end=-1): return nn.Sequential(*list(model.children())[beginning:end])
def get_children(model): return list(model.children())
def get_all_children(model, children = []):
    for child in model.children():
        children.append(child)
        get_all_children(child, children)
    return children
def get_terminal_children(model, res=None):
    if res is None: res = []
    children = list(model.children())
    if not children: res.append(model)
    else:
        for child in children: get_terminal_children(child, res)
    return res

def get_size_changes(children, allow_duplicates = False):
    #Add the first module as well
    in_c, out_c = getattr(children[0], 'in_channels', None), getattr(children[0], 'out_channels', None)
    modules = [(in_c, out_c, children[0])] if (in_c and out_c) else []
    sizes = set()
    for module in children:
        in_c, out_c = getattr(module, 'in_channels', None), getattr(module, 'out_channels', None)
        if (in_c and out_c) and (in_c != out_c):
            if allow_duplicates or (in_c not in sizes): modules.append((in_c, out_c, module))
            sizes.add(in_c)
    return modules

def record_input(hook, module, inp, out):
    hook.inp = inp[0] if isinstance(inp, tuple) else inp

class Hook():
    def __init__(self, module, hook_function):
        self.module, self.hook_function = module, hook_function
        self.hook = module.register_forward_hook(partial(hook_function, self))
    def remove(self):
        self.hook.remove()
    def __del__(self): self.remove()
    def __repr__(self): return f"Hook for {self.module}"

class UNetHooks():
    def __init__(self, mods, hook_function=record_input):
        self.mods, self.hook_function = mods, hook_function
        self.register_hooks()
    def register_hooks(self):
        if hasattr(self, 'hooks'): self.remove()
        self.hooks = [Hook(module, self.hook_function) for module in self.mods]
    def remove(self):
        for h in self.hooks: h.remove()
    def __del__(self): self.remove()
    def __getitem__(self, i): return self.hooks[i]
    def add_module(self, module): self.hooks.append(Hook(module, self.hook_function))
    def __repr__(self): return f"Hook container with {len(self)} Hooks"
    def __len__(self): return len(self.hooks)

class UnetBlock(nn.Module):
    def __init__(self, hook, up_in_c, x_in_c, blur=None,leaky=None):
        """
        up_in_c - The number of input channels from the layer below
        x_in_c  - The number of input channels coming in from the hook
        hook    - a hook that retrieves the output from the other side of the network
        """
        super().__init__()
        self.hook = hook
        self.upsampler = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, leaky=leaky)
        ni = up_in_c//2 + x_in_c
        nf = ni
        self.bn = nn.BatchNorm2d(x_in_c)
        with torch.no_grad():
            self.bn.bias.fill_(1e-3)
            self.bn.weight.fill_(1.)
        self.blur = False
        self.conv1 = conv_layer(ni, nf, leaky=leaky, norm_type=None)
        self.conv2 = conv_layer(nf, nf, leaky=leaky, norm_type=None)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, up_in):
        x_in = self.hook.inp
        up_out = self.upsampler(up_in)
        if up_out.shape[-2:] != x_in.shape[-2:]:
            up_out = F.interpolate(up_out, x_in.shape[-2:])
        x = self.relu(torch.cat([up_out, self.bn(x_in)], dim=1))
        return self.conv2(self.conv1(x))

    def __del__(self): self.hook.remove()

def get_upsample_inputs(inp_sizes, out_sizes):
    upsample_inputs = [out_sizes[-1]]

    for inp_size in reversed(inp_sizes):
        #print(inp_size, upsample_inputs[-1])
        upsample_inputs.append(inp_size + upsample_inputs[-1]//2)
    return upsample_inputs

class Head(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = conv_layer(n_in, n_out, stride = 2)
        self.relu = nn.LeakyReLU(0.01)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, x): return self.relu(self.bn(self.conv(x)))

class Encoder(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.n_in = n_in
        self.bn = nn.BatchNorm2d(n_in)
        self.conv1 = conv_layer(n_in, n_in*2)
        self.conv2 = conv_layer(n_in*2, n_in)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x): return self.relu(self.conv2(self.relu(self.conv1(self.bn(x)))))

class Tail(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.n = n_in
        self.conv1 = conv_layer(n_in, n_in)
        self.conv2 = conv_layer(n_in, n_in)
        self.conv_final = conv_layer(n_in,n_out, ks=1)
        self.relu = nn.LeakyReLU(0.01)

    def forward(self, x): return self.conv_final(self.relu(self.conv2(self.relu(self.conv1(x)))))


class UNet(nn.Module):
    def __init__(self, c_in, c_out, up_in_c, base_model, unet_blocks, head, encoder, tail):
        super().__init__()
        self.c_in, self.c_out, self.base_model = c_in, c_out, base_model
        self.head, self.encoder, self.tail, self.unet_blocks = head, encoder, tail, unet_blocks
        self.upsampler = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=None, leaky=None)

    def forward(self, x):
        x_head = self.head(x)
        x_body = self.base_model(x_head)
        x_encoder = self.encoder(x_body)
        x_upscale = self.unet_blocks(x_encoder)
        x_concat = torch.cat([x, self.upsampler(x_upscale)], dim=1)
        return self.tail(x_concat)


class DynamicResnet34Unet(nn.Module):
    def __init__(self, c_in, c_out, base_model, head=Head, encoder=Encoder, tail=Tail):
        super().__init__()
     
        self.base_model= cut_model(base_model, 4, -2)
        size_change_array = get_size_changes(get_terminal_children(self.base_model), allow_duplicates=False)
        inp_sizes, out_sizes, mods = map(list, zip(*size_change_array))
        self.mods = mods
        self.unet_hooks = UNetHooks(mods)
        self.upsample_inputs = get_upsample_inputs(inp_sizes, out_sizes)
        self.unet_components = [UnetBlock(hook, up_in_c, x_in_c) for
               hook, up_in_c, x_in_c in zip(reversed(self.unet_hooks), self.upsample_inputs, reversed(inp_sizes))]
        self.unet_blocks = nn.Sequential(*self.unet_components)
        self.head = head(c_in, inp_sizes[0])

        self.encoder = encoder(out_sizes[-1])
        self.tail    = tail(self.upsample_inputs[-1]//2+c_in, c_out)
        self.upsampler = PixelShuffle_ICNR(self.upsample_inputs[-1], self.upsample_inputs[-1]//2)

    def forward(self, x):
        x_head = self.head(x)
        x_body = self.base_model(x_head)
        x_encoder = self.encoder(x_body)
        x_upscale = self.unet_blocks(x_encoder)
        x_concat = torch.cat([x, self.upsampler(x_upscale)], dim=1)
        return self.tail(x_concat)

    def __del__(self): self.unet_hooks.remove()

