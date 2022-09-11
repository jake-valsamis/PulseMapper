
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
# file to edit: dev_nb/Learner.ipynb
import torch
import numpy as np
import re
import math
import random
from functools import partial
import matplotlib.pyplot as plt
import imageio
from pathlib import Path

from utils import listify, compose, setify

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

class Callback():
    """
    _order - the key used to order the various callbacks associated with your learner (lower are called earlier)

    When a callback is called (e.g. self('begin_batch'), the class instance will check whether it has a class
    method with that name (e.g. def begin_batch).  If it exists, it will call that method and return the results,
    otherwise it will simply return False

    For this application, False is considered the default response, since calling any of these callbacks will return
    False unless otherwise defined.
    """
    _order = 0
    def set_learner(self, learn): self.learn=learn
    def __getattr__(self, k): return getattr(self.learn, k)  #Will look in self.learn if it can't otherwise find the attr.
    @property
    def name(self): return camel2snake(re.sub('Callback$', '', self.__class__.__name__)) or 'callback'

    def __call__(self, cb_name):
        #call the specific function if it has been defined by the subclass, otherwise return False
        if hasattr(self, cb_name): return getattr(self, cb_name)()
        else: return False

class TrainEvalCallback(Callback):
    """ A mandatory callback for the Learner class (could be built in directly as well)
    Handles some of the basic tasks associated with keeping track of the iteration number, etc.
    as well as controlling the model modes (train vs eval).
    """
    def begin_fit(self):
        self.learn.n_epochs=0
        self.learn.n_iter=0
    def after_batch(self):
        self.learn.n_epochs += 1./self.iters
        self.learn.n_iter += 1
    def begin_epoch(self):
        self.learn.n_epochs = self.epoch
        self.model.train()
        self.learn.in_train=True
    def begin_validate(self):
        self.model.eval()
        self.learn.in_train=False

def param_getter(m): return m.parameters()
class CancelBatchException(Exception): pass
class CancelEpochException(Exception): pass
class CancelTrainException(Exception): pass


class Learner():
    def __init__(self, model, data, loss_func, opt_func, lr=1e-2, splitter=param_getter, cbs=None, cbfs=None):
        """
        model     - a PyTorch model (based on nn.Modules or nn.Sequential)
        data      - a class containing the train_dl and valid_dl dataloaders (each has an iterator that outputs xb,yb)
        loss_func - the function used to calculate your loss during training
        opt_func  - a function that will create your optimizer.  Usually a partial function to create the Optimizer class
        lr        - the maximum learning rate for your system (may be scaled using the parameter schedulers)
        splitter  - the function to get all of the parameters of your network and arrange them into param_groups
        cbs       - a list of instances of the Callback-related classes
        cbfs      - a list of functions that can be used to create Callback instances

        Notes about the Learner:
        Control is handled using exceptions, which can be raised by function or callbacks.
        The three exceptions are:
        CancelBatchException - Will cancel the current batch, but continue on with the next batch
        CancelEpochException - Will cancel the current epoch, but continue on with the next epoch
        CancelTrainException - Will cancel the current training cycle
        When any of the exceptions is raised, the associated callback functions (e.g. after_cancel_epoch) will be run
        All other exceptions will trigger as normal
        """
        self.model,self.data,self.loss_func,self.opt_func,self.lr,self.splitter=model,data,loss_func,opt_func,lr,splitter
        #Initialize other parameters.  Logger determines how information is presented (in this case printed out)
        self.in_train,self.logger,self.opt = False,print,None

        #Initilize all the callbacks.  This must be done properly since each callback must be associated with the learner
        #This allows it to access all the attributes of the learner
        self.cbs = []
        self.add_cbs(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cbfs))

    #Functions required to process callbacks
    ALL_CB_TYPES = {'begin_batch', 'after_pred', 'after_loss', 'after_backward', 'after_step',
        'after_cancel_batch', 'after_batch', 'after_cancel_epoch', 'begin_fit',
        'begin_epoch', 'begin_validate', 'after_epoch',
        'after_cancel_train', 'after_fit'}
    def add_cbs(self, cbs):
        for cb in listify(cbs): self.add_cb(cb)
    def remove_cbs(self, cbs):
        for cb in listify(cbs): self.cbs.remove(cb)
    def add_cb(self, cb):
        cb.set_learner(self)        #associate the callback with this learner
        setattr(self, cb.name, cb)  #allow the learner to access the callback by name
        self.cbs.append(cb)
    def __call__(self, cb_name):
        #The boolean aspect is only used for determining whether to go through with an epoch or not and could be
        #refactored out in the future
        res = False
        assert cb_name in self.ALL_CB_TYPES, f'callback {cb_name} not found in callbacks:\,{self.ALL_CB_TYPES}'
        for cb in sorted(self.cbs, key=lambda x:x._order):  res = cb(cb_name) and res
        return res


    #Functions that control training
    def fit(self, epochs, cbs=None, reset_opt=False):
        """Begin training
        epochs    - number of epochs that you want to train
        cbs       - any additional callbacks that you want to add during training and removed afterwards
        reset_ops - determine whether you want to reset the optimizer (e.g. if it was holding on to momentum, etc.)
        """
        self.add_cbs(cbs)
        if reset_opt or self.opt is None: self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)

        try:
            self.epochs, self.loss = epochs, torch.tensor(0.);   self('begin_fit')
            for epoch in range(epochs):
                print(f"epoch#{epoch}\n")
                self.epoch, self.dl = epoch, self.data.train_dl
                if not self('begin_epoch'): self.all_batches()

                with torch.no_grad():
                    self.dl = self.data.valid_dl
                    if not self('begin_validate'): self.all_batches()
                self('after_epoch')

        except CancelTrainException: self('after_cancel_train')
        except Exception as e: raise e
        finally:
            self('after_fit')
            self.remove_cbs(cbs)

    def all_batches(self):
        self.iters = len(self.dl)
        try:
            for i, (xb, yb) in enumerate(self.dl): self.one_batch(i, xb, yb)
        except CancelEpochException: self('after_cancel_epoch')
        except Exception as e: raise e

    def one_batch(self, i, xb, yb):
        """Process one batch of data (either train or valid)
        i   - The iteration number of this epoch
        xb  - A single batch of inputs  as output by your dataloader.
        yb  - A single batch of outputs as output by your dataloader
        """
        try:
            self.iter = i
            self.xb, self.yb = xb, yb;                        self('begin_batch')
            self.pred = self.model(self.xb);                  self('after_pred')
            self.loss = self.loss_func(self.pred, self.yb);   self('after_loss')
            if not self.in_train: return
            self.loss.backward();                             self('after_backward')
            self.opt.step();                                  self('after_step')
            self.opt.zero_grad()
        except CancelBatchException:                          self('after_cancel_batch')
        except Exception as e: raise e
        finally: self('after_batch')


    def show_results(self, n=100, idxs=None, valid=True):

        self.in_train=False
        dl=self.data.valid_dl if valid else self.data.train_dl
        xb,yb=next(iter(dl)) if idxs is None else dl.custom_batch(idxs)
        self.one_batch(0,xb,yb)
        xb,yb,pred=self.xb,self.yb,self.pred

        xb, yb, pred = map(torch_to_numpy, (xb,yb,pred))
        xb, yb, pred = map(normalize_channels, (xb,yb,pred))
        #xb, yb, pred = map(transpose_for_plotting, (xb, yb, pred))

        bs,c,h,w=xb.shape
        rows=min(n,bs)
        SIZE=100/rows
        fig,axes=plt.subplots(rows,3,figsize=(SIZE,SIZE*rows/3))
        for j,title in enumerate(['Input', 'Prediction','Ground Truth']):
            axes[0][j].set_title(title, fontsize=SIZE*1.5)
        for i in range(rows):
            for j,img_stack in enumerate((xb, pred, yb)):
                axes[i][j].axis('off')
                axes[i][j].imshow(transpose_for_plotting(img_stack[i,...]))
        fig.tight_layout()
        return fig


def torch_to_numpy(x): return np.array(x.detach().cpu()) if isinstance(x, torch.Tensor) else x

def normalize_channels(x):
    x-= np.min(x, axis=(-2,-1), keepdims=True)
    x/= np.max(x, axis=(-2,-1), keepdims=True)
    return x
def transpose_for_plotting(x):
    ndims=len(x.shape)
    #axes = tuple((i+1)%ndims for i in range(ndims))
    axes=(0,2,3,1) if ndims==4 else (1,2,0)
    x= np.transpose(x, axes=axes)
    if x.shape[-1]==1: x=x.squeeze(-1)
    elif x.shape[-1]==2:x=np.concatenate((x, np.zeros(list(x.shape[:-1])+[1])), axis=-1)
    elif x.shape[-1]>3:x=np.concatenate((x[...,:2],x[...,2:].mean(axis=-1,keepdims=True)),axis=-1)
    return x


class ToDeviceCallback(Callback):
    _order=10
    def __init__(self, device):
        self.device = device
    def begin_fit(self): self.learn.model = self.learn.model.to(self.device)
    def begin_batch(self):
        #if not isinstance(self.learn.xb, torch.Tensor): self.learn.xb = torch.tensor(self.learn.xb)
        #if not isinstance(self.learn.yb, torch.Tensor): self.learn.yb = torch.tensor(self.learn.yb)
        self.learn.xb = self.learn.xb.to(self.device)
        self.learn.yb = self.learn.yb.to(self.device)

class ToFloatCallback(Callback):
    _order = 1
    def __init__(self, do_x=True, do_y=True): self.do_x, self.do_y=do_x, do_y
    def begin_batch(self):
        if self.do_x: self.learn.xb = self.learn.xb.float()
        if self.do_y: self.learn.yb = self.learn.yb.float()


class AvgStats():
    def __init__(self, metrics, in_train): self.metrics, self.in_train = listify(metrics), in_train
    def reset(self):
        self.tot_loss, self.count = 0., 0
        self.tot_metrics = [0.]*(len(self.metrics))
    def __repr__(self): return f"{'Train' if self.in_train else 'Valid:'} {self.avg_stats}"
    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_metrics
    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]
    def accumulate(self, learn):
        batch_size = learn.xb.shape[0]
        self.tot_loss += learn.loss * batch_size
        self.count += batch_size
        for i, metric in enumerate(self.metrics):
            self.tot_metrics[i] += metric(learn.pred, learn.yb) *batch_size


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats, self.valid_stats = AvgStats(metrics, True), AvgStats(metrics, False)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()

    def after_loss(self):
        current_stats = self.train_stats if self.learn.in_train else self.valid_stats
        with torch.no_grad(): current_stats.accumulate(self.learn)

    def after_epoch(self):
        print(self.train_stats)
        print(self.valid_stats)



class Printer(Callback):
    #def begin_batch(self): print(self.learn.xb.shape)
    def after_batch(self): print (f"Finished batch {self.learn.iter}")
    def begin_validate(self): print('Begin validate')

def standardize(x, m, s): return (x-m)/s

def stats_by_channel(arr, axis=(1,2), keepdim=True):
    n = len(arr)
    mean = torch.zeros_like(torch.mean(arr[0].data, axis=axis, keepdim=keepdim))
    std = torch.zeros_like(mean)
    for i in range(n):
        mean += torch.mean(arr[i].data, axis=axis, keepdim=keepdim)
    mean/=n
    for i in range(n):
        std += torch.mean((arr[i].data-mean)**2, axis=axis, keepdim=keepdim)
    std = (std/n).sqrt()
    return mean,std

def stats_by_channel_generic(arr, axis=(1,2), keepdim=True):
    n = len(arr)
    mean = torch.zeros_like(torch.mean(arr[0], axis=axis, keepdim=keepdim))
    std = torch.zeros_like(mean)
    for i in range(n):
        mean += torch.mean(arr[i].data, axis=axis, keepdim=keepdim)
    mean/=n
    for i in range(n):
        std += torch.mean((arr[i].data-mean)**2, axis=axis, keepdim=keepdim)
    std = (std/n).sqrt()
    return mean,std

class StandardizeBatchCallback(Callback):
    _order = 5
    def __init__(self,x_mean=None, x_std=None, y_mean=None, y_std=None, stats_func = stats_by_channel):
        self.x_mean, self.y_mean, self.x_std, self.y_std = x_mean, y_mean, x_std, y_std
        self.stats_func = stats_func
    def begin_fit(self):
        if self.x_mean is None or self.x_std is None: self.x_mean, self.x_std = self.stats_func(self.learn.data.train_ds.x)
        if self.y_mean is None or self.y_std is None: self.y_mean, self.y_std = self.stats_func(self.learn.data.train_ds.y)
    def begin_batch(self):
        xtype, ytype = self.learn.xb.dtype, self.learn.yb.dtype
        device = self.learn.xb.device
        self.learn.xb = standardize(self.learn.xb, self.x_mean.type(xtype).to(device), self.x_std.type(xtype).to(device))
        self.learn.yb = standardize(self.learn.yb, self.y_mean.type(ytype).to(device), self.y_std.type(ytype).to(device))

class StandardizeXBatchCallback(Callback):
    _order = 5
    def __init__(self,x_mean=None, x_std=None, y_mean=None, y_std=None, stats_func = stats_by_channel):
        self.x_mean, self.y_mean, self.x_std, self.y_std = x_mean, y_mean, x_std, y_std
        self.stats_func = stats_func
    def begin_fit(self):
        if self.x_mean is None or self.x_std is None: self.x_mean, self.x_std = self.stats_func(self.learn.data.train_ds.x)
#         if self.y_mean is None or self.y_std is None: self.y_mean, self.y_std = self.stats_func(self.learn.data.train_ds.y)
    def begin_batch(self):
        xtype, ytype = self.learn.xb.dtype, self.learn.yb.dtype
        device = self.learn.xb.device
        self.learn.xb = standardize(self.learn.xb, self.x_mean.type(xtype).to(device), self.x_std.type(xtype).to(device))
#         self.learn.yb = standardize(self.learn.yb, self.y_mean.type(ytype).to(device), self.y_std.type(ytype).to(device))


def normalize_by_channel(x):
    bs,c,h,w=x.shape
    x = x.view(x.size(0),x.size(1), -1)
    x -= x.min(2, keepdim=True)[0]
    x /= x.max(2, keepdim=True)[0]
    x = x.view(bs,c, h, w)
    return x


class NormalizeBatchCallback(Callback):
    _order = 5
    def __init__(self, norm_x=False, norm_y=False):
        self.norm_x, self.norm_y=norm_x,norm_y
        if not norm_x and not norm_y:
            print("""Warning: you have included NormalizeBatch but are not normalizing the inputs or label
                  Set either norm_x or norm_y to True""")
    def begin_batch(self):
        if self.norm_x:self.learn.xb=normalize_by_channel(self.learn.xb)
        if self.norm_y:self.learn.yb=normalize_by_channel(self.learn.yb)



####  ©nandinee
class Recorder(Callback):
    def begin_fit(self): self.lrs,self.losses, self.epoch_lrs, self.epoch_losses=[],[],[],[]
    def after_batch(self):
        if not self.in_train:return
        self.lrs.append(self.learn.opt.hypers[-1]['lr'])
        self.losses.append(self.learn.loss.detach().cpu())
    ####  ©nandinee
    def after_epoch(self):
        if not self.in_train:return
        self.epoch_lrs.append(self.learn.opt.hypers[-1]['lr'])
        self.epoch_losses.append(self.learn.loss.detach().cpu())
    def plot_losses(self):plt.plot(self.losses)
    def plot_lrs(self):plt.plot(self.lrs)
    ####  ©nandinee
    def plot_epoch_losses(self):plt.plot(self.epoch_losses)
    def plot_epoch_lrs(self):plt.plot(self.epoch_lrs)
    def plot(self, skip_last=0):
        losses=[o.item() for o in self.losses]
        n=len(losses)-skip_last
        ax.set_xscale('log')
        plt.plot(self.lrs[:n],losses[:n])


####  ©nandinee
class Recorder(Callback):
    def begin_fit(self): self.lrs,self.losses, self.epoch_lrs, self.epoch_losses=[],[],[],[]

    def after_batch(self):
        if not self.in_train: return
        self.lrs.append(self.opt.hypers[-1]['lr'])
        self.losses.append(self.loss.detach().cpu())
    ####  ©nandinee
    def after_epoch(self):
        if not self.in_train:return
        self.epoch_lrs.append(self.opt.hypers[-1]['lr'])
        self.epoch_losses.append(self.loss.detach().cpu())
    def plot_lr  (self):
        fig, ax = plt.subplots()
        ax.plot(self.lrs)
        return fig
    def plot_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self.losses)
        return fig
    ####  ©nandinee
    def plot_epoch_lr(self):
        fig, ax = plt.subplots()
        ax.plot(self.epoch_lrs)
        return fig
    def plot_epoch_loss(self):
        fig, ax = plt.subplots()
        ax.plot(self.epoch_losses)
        return fig
    def plot(self, skip_last=0):
        losses = [o.item() for o in self.losses]
        n = len(losses)-skip_last
        fig, ax = plt.subplots()
        ax.set_xscale('log')
        ax.plot(self.lrs[:n], losses[:n])
        return fig
class LR_Find(Callback):
    _order=1
    def __init__(self, max_iter=100,min_lr=1e-6,max_lr=10):
        self.max_iter,self.min_lr,self.max_lr=max_iter,min_lr,max_lr
        self.best_loss=1e9
    def begin_batch(self):
        if not self.learn.in_train:return
        pos=self.n_iter/self.max_iter
        lr=self.min_lr*(self.max_lr/self.min_lr)**pos
        for pg in self.learn.opt.hypers:pg['lr']=lr

    def after_batch(self):
        if self.n_iter>self.max_iter or self.learn.loss>self.best_loss*10: raise CancelTrainException()
        if self.learn.loss<self.best_loss:self.best_loss=self.learn.loss



class BatchTransformCallback(Callback):
    _order=5
    def __init__(self, tfms, shuffle = True, stack_x:bool=False, stack_y:bool=True, tfm_train=True):
        self.tfms = tfms
        self.shuffle = shuffle
        self.stack_x,self.stack_y = stack_x,stack_y
        self.tfm_train=tfm_train

    def begin_batch(self):
        if not self.in_train or not self.tfm_train: return
        tfms = self.tfms
        size=self.learn.xb.shape[-1]
        if self.shuffle: random.shuffle(tfms)
        for tsfm_func, p, max_kwargs in tfms:
            if p>random.random():
                new_kwargs = {name:choose_arguments(value) for name, value in max_kwargs.items()}
                self.learn.xb,self.learn.yb=map(lambda x:tsfm_func(x, size, **new_kwargs),(self.learn.xb, self.learn.yb))


def randomize_arguments(value):
    if isinstance(value, bool): return value
    elif isinstance(value, int) or isinstance(value, float): return value*np.random.randn()
    else: return value

def choose_arguments(value):
    """If you provide a tuple, choose a uniform value between them (may change to standard distribution)"""
    if isinstance(value, tuple):
        low, high = value
        return low + (high-low)*np.random.random()
    else: return value

def annealer(f):
    def _inner(start,end):return partial(f,start,end)
    return _inner

@annealer
def sched_cos(start,end,pos):return start+(1+math.cos(math.pi*(1-pos)))*(end-start)/2
@annealer
def sched_lin(start,end,pos):return start+(end-start)*pos
@annealer
def sced_no(start,end,pos):return start
@annealer
def sched_exp(start,end,pos):return start*(end/start)**pos

def combine_scheds(pcts,scheds):
    assert sum(pcts)==1.
    pcts=torch.tensor([0.]+listify(pcts))
    assert torch.all(pcts>=0)
    pcts=torch.cumsum(pcts,0)
    def _inner(pos):
        idx=(pos>=pcts).nonzero().max()
        actual_pos=(pos-pcts[idx])/(pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos)
    return _inner


class ParamScheduler(Callback):
    _order=1
    def __init__(self,pname,sched_func):self.pname,self.sched_func=pname,listify(sched_func)
    def begin_batch(self):
        if not self.in_train: return
        fs=self.sched_func
        if len(fs)==1:fs=fs*len(self.learn.opt.param_groups)
        pos = self.learn.n_epochs/self.learn.epochs
        for f,hyper in zip(fs,self.learn.opt.hypers):hyper[self.pname]=f(pos)


def get_file_subpath(fn, ext='.npy'):
    slice_name = fn.name.replace(ext,'')
    image_name = fn.parent.parent.name
    return Path(image_name)/slice_name

def get_norm_stats_by_channels(x):
    sub = np.min(x, axis=(-2,-1), keepdims=True)
    div = np.max(x-sub, axis=(-2,-1), keepdims=True)
    return sub, div

class OutputResultsCallback(Callback):
    _order=1
    def __init__(self, output_folder, output_raw=True): 
        self.output_folder=output_folder
        self.output_raw = output_raw
    def begin_fit(self):
        self.temp_tfms = self.learn.batch_transform.tfms
        self.learn.batch_transform.tfms = {}
    def after_pred(self):
        subfolders = [get_file_subpath(x[0].fn) for x in self.learn.dl.batch]
        xb,yb,pred=self.xb,self.yb,self.pred
        xb, yb, pred = map(torch_to_numpy, (xb,yb,pred))
        n = xb.shape[0]
        if self.output_raw:
            for i in range(n):
                save_path = self.output_folder/subfolders[i]
                save_path.parent.mkdir(exist_ok=True, parents=True)
                for suffix, stack in [('-x',xb),('-y',yb),('-pred',pred)]:
                    np.save(save_path.parent/f"{save_path.name}{suffix}.npy", stack[i,...])
        
        
        xb, yb, pred = map(np.abs, (xb,yb,pred))

        #Note: May be better to normalize yb and pred using the same scalars
        sub, div = get_norm_stats_by_channels(yb)
        pred_alt = np.clip((pred-sub)/div, 0, 1)


        xb, yb, pred = map(normalize_channels, (xb,yb,pred))
        #xb, yb, pred = map(lambda x: (x*255).astype('uint8'), (xb,yb,pred))

#         xb = normalize_channels(xb)
#         sub, div = get_norm_stats_by_channels(yb)
#         yb = (yb-sub)/div
#         pred = np.clip((pred-sub)/div, 0, 1)


        n = xb.shape[0]
        for i in range(n):
            save_path = self.output_folder/subfolders[i]
            save_path.parent.mkdir(exist_ok=True, parents=True)
            for suffix, stack in [('-x',xb),('-y',yb),('-pred',pred), ('-predyscale', pred_alt)]:
                output_image = (transpose_for_plotting(stack[i,...])*255).astype('uint8')
                imageio.imsave(save_path.parent/f"{save_path.name}{suffix}.jpg", output_image)
    def after_loss(self): raise CancelBatchException()
    def after_fit(self): self.learn.batch_transform.tfms = self.temp_tfms




# class ModelSaverCallback(Callback):
#     def __init__(self, save_after_epochs=1):
#         self.save_after_epochs=save_after_epochs

#     def after_epoch(self):
#         torch.save(self.learn.model, self.output_folder)