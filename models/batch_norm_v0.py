"""Mecta Batch-norm with simplified implementation.
Note the implementation does not really reduce memory and may include more memory consumption than calculated cache sizes.
"""
import numpy as np
from copy import deepcopy
import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d


class MectaNorm2d(BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True,
                 accum_mode='exp', beta=0.1, use_forget_gate=False,
                 init_beta=None,
                 verbose=False, dist_metric='skl', bn_dist_scale=1.,
                 beta_thre=0., prune_q=0., name='bn',
                 ):
        super(MectaNorm2d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats)
        # self.train(False)
        self.name = name
        self.accum_mode = accum_mode
        self.beta = beta
        self.init_beta = self.beta if init_beta is None else init_beta
        self.past_size = 0
        self.use_forget_gate = use_forget_gate

        self.verbose = verbose

        self.bn_dist_scale = bn_dist_scale
        # self.var_debias = True

        self.update_dist_metric(dist_metric)  # simple | kl | skl

        self.beta_thre = beta_thre
        self.full_matched = False
        self.prune_q = prune_q
        # self.prune_mode = 'lamp'

        if self.prune_q > 0.:
            self.mask = torch.ones_like(self.weight)

            def backward_hook(grad):
                out = grad.clone()
                if self.mask is not None:
                    out = out * self.mask  # mask out gradients.
                # print(f"## kept rate at grad: {self.mask.mean()}")
                return out

            self.weight.register_hook(backward_hook)
        else:
            self.mask = None

        self.forward_cache_size = None

    # def update_accum_params(self, accum_mode=None, beta=None, use_forget_gate=None,
    #                         init_beta=None,
    #                         verbose=None, dist_metric=None, bn_dist_scale=None):
    #     if accum_mode is not None:
    #         self.accum_mode = accum_mode
    #     if beta is not None:
    #         self.beta = beta
    #     if use_forget_gate is not None:
    #         self.use_forget_gate = use_forget_gate
    #     if init_beta is not None:
    #         self.init_beta = init_beta
    #     if verbose is not None:
    #         self.verbose = verbose
    #     if dist_metric is not None:
    #         self.update_dist_metric(dist_metric)
    #     if bn_dist_scale is not None:
    #         self.bn_dist_scale = bn_dist_scale
    #     # if var_debias is not None:
    #     #     self.var_debias = var_debias

    def update_dist_metric(self, dist_metric):
        if dist_metric == 'kl':
            self.dist_metric = gauss_kl_divergence
        elif dist_metric == 'skl':
            self.dist_metric = gauss_symm_kl_divergence
        elif dist_metric == 'simple':
            self.dist_metric = simple_divergence
        elif dist_metric == 'mmd':
            self.dist_metric = mmd_divergence
        else:
            raise RuntimeError(f"Unknown distance: {dist_metric}")

    def reset(self):
        # if not keep_stats:
        #     self.reset_running_stats()
        self.past_size = 0
        self.forward_cache_size = None
        self.full_matched = False

    def cache_size(self):
        assert self.forward_cache_size is not None, "Please call this after forward!"
        forward_cache = self.forward_cache_size
        backward_cache = 0.
        if self.weight.requires_grad and not self.full_matched:
            if self.mask is not None:
                backward_cache = forward_cache[0] * self.mask.sum(
                ).item() * forward_cache[2] * forward_cache[3]
            else:
                backward_cache = np.prod(forward_cache)
            backward_cache *= dtype_memory_size_dict[torch.float]
        forward_cache = np.prod(forward_cache) * \
            dtype_memory_size_dict[torch.float]
        # else:
        return forward_cache, backward_cache

    def set_accum_mode(self, accum_mode):
        self.accum_mode = accum_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_cache_size = x.shape
        if self.accum_mode == 'exp':  # suitable for batch streaming
            x_size = len(x)

            batch_mean = torch.mean(x, dim=(0, 2, 3))
            batch_var = torch.var(x, dim=(0, 2, 3), unbiased=False)
            current_mean = batch_mean
            current_var = batch_var

            if self.use_forget_gate:
                # if self.past_size == 0:  # store init stat
                #     self.train_running_mean = deepcopy(self.running_mean.data.detach())
                #     self.train_running_var = deepcopy(self.running_var.data.detach())

                if self.running_mean is None:
                    raise RuntimeError(f"Not found running_mean.")
                if self.running_var is None:
                    raise RuntimeError(f"Not found running_var.")
                if self.use_forget_gate:
                    dist = self.dist_metric(current_mean, current_var,
                                            self.running_mean, self.running_var)
                    mean_dist = torch.mean(dist)

                    forget_gate = torch.exp(- self.bn_dist_scale * mean_dist)

                    forget_gate = forget_gate.detach()
                    # if self.verbose:
                    #     print("forget_gate", forget_gate.item())
                else:
                    forget_gate = 0.

                beta = 1. - forget_gate
                # for the purpose of logging.
                self.beta = beta.item() if hasattr(beta, 'item') else beta
                # FIXME ad-hoc detach the current mean for preserving grad. Refer to Yang2022 Test-time BN.
                current_mean = beta * current_mean.detach()
                current_var = beta * current_var.detach()

                if (forget_gate > 0.):
                    current_mean = current_mean + forget_gate * self.running_mean.detach()
                    # + beta * (1. - beta) * (current_mean - self.running_mean.detach()) ** 2
                    current_var = current_var + forget_gate * self.running_var.detach()

                self.running_mean.data.copy_(current_mean.detach())
                self.running_var.data.copy_(current_var.detach())
            else:  # use constant beta
                beta = self.beta if self.past_size > 0 else self.init_beta

                if beta < 1.:  # accumulate
                    current_mean = beta * current_mean.detach()
                    current_var = beta * current_var.detach()

                    current_mean = current_mean + \
                        (1-beta) * self.running_mean.detach()
                    current_var = current_var + \
                        (1-beta) * self.running_var.detach()

                    self.running_mean.data.copy_(current_mean.detach())
                    self.running_var.data.copy_(current_var.detach())

            # if remove this, make sure the current_mean/var will BP.
            x = PreG_norm(x, batch_mean, batch_var, ignore_var=False)

            if beta > self.beta_thre:
                self.full_matched = False
                weight, bias = self.weight, self.bias
            else:  # stop grad
                self.full_matched = True
                # print(f"### no cache for {self.name}")
                # detach to avoid cache and grad.
                weight, bias = self.weight.detach(), self.bias.detach()
            if self.prune_q > 0. and self.weight.requires_grad:
                self.mask = torch.ones_like(self.weight)
                n_rm = int(len(self.weight) * self.prune_q)
                idxs = torch.randperm(len(weight))
                self.mask[idxs[:n_rm]] = 0.
            y = batch_norm(x, current_mean, current_var, weight, bias)

            # update stat
            self.past_size += x_size
        elif self.accum_mode == 'none':  # standard BN
            y = super(MectaNorm2d, self).forward(x)
        else:
            raise NotImplementedError(f"accum_mode: {self.accum_mode}")

        return y


class standard_bn_cxt:
    """Make AccumBN into standard BN within context.
    Use this to disable adaptation.
    1. Set accum mode to none.
    2. Remove grad mask.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.accum_modes = {}

    def __enter__(self):
        self.accum_modes = {}
        for name, m in self.model.named_modules():
            if isinstance(m, MectaNorm2d):
                self.accum_modes[name] = m.accum_mode
                m.set_accum_mode('none')
                m.mask = None  # avoid grad masking.

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, m in self.model.named_modules():
            if isinstance(m, MectaNorm2d):
                m.set_accum_mode(self.accum_modes[name])
        self.accum_modes = {}


def get_bn_cache_size(model: nn.Module):
    max_forward_cs = 0.
    total_backward_cs = 0.
    for m in model.modules():
        if isinstance(m, MectaNorm2d):
            forward_cs, backward_cs = m.cache_size()
            if forward_cs > max_forward_cs:
                max_forward_cs = forward_cs
            total_backward_cs += backward_cs
    return max_forward_cs, total_backward_cs


def simple_divergence(mean1, var1, mean2, var2):
    return (mean1 - mean2) ** 2 / var2


def gauss_kl_divergence(mean1, var1, mean2, var2):
    # /// v1: relative to distribution 2 ///
    d1 = (torch.log(var2) - torch.log(var1))/2. + \
        (var1 + (mean1 - mean2)**2) / 2. / var2 - 0.5
    return d1


def gauss_symm_kl_divergence(mean1, var1, mean2, var2):
    # /// v2: symmetric ///
    # d1 = (torch.log(var2) - torch.log(var1))/2. + (var1 + (mean1 - mean2)**2) / 2. / var2 - 0.5
    # d2 = (torch.log(var1) - torch.log(var2)) / 2. + (var2 + (mean1 - mean2) ** 2) / 2. / var1 - 0.5
    # return d1 + d2
    # equivalent to
    d1 = (var1 + (mean1 - mean2) ** 2) / 2. / var2 - 0.5
    d2 = (var2 + (mean1 - mean2) ** 2) / 2. / var1 - 0.5
    return d1 + d2


def mmd_divergence(mean1, var1, mean2, var2):
    d1 = torch.sqrt((var1 - var2) ** 2 + (mean1 - mean2) ** 2)
    return d1


def PreG_norm(x, current_mean, current_var, ignore_var=False):
    """Do normalization while preserving the back-propagated gradients.
    current_mean, current_var should not stop grad,"""
    current_mean = current_mean.view((1, -1, 1, 1))
    y = x - current_mean

    if not ignore_var:
        current_var = current_var.view((1, -1, 1, 1))
        y = y / torch.sqrt(current_var) * torch.sqrt(current_var.detach())

    y = y + current_mean.detach()
    return y


def batch_norm(x, current_mean, current_var, weight, bias):
    """BN enabling BP using given stats."""
    x_norm = (x - current_mean.reshape((1, -1, 1, 1))) / torch.sqrt(current_var).reshape(
        (1, -1, 1, 1))
    # re-norm refer to 'Online Normalization for Training Neural Networks'
    # x_norm = x_norm / torch.sqrt(torch.mean(x_norm**2, dim=(0,2,3), keepdim=True))
    # print(f"$$$ self.weight {self.weight.shape} self.bias {self.bias.shape}")
    y = x_norm * weight.reshape((1, -1, 1, 1)) + bias.reshape((1, -1, 1, 1))
    return y


def get_last_beta(model: nn.Module):
    all_betas = []
    for m in model.modules():
        if isinstance(m, MectaNorm2d):
            all_betas.append(m.beta)
    return all_betas


def has_accum_bn_grad(model):
    """Return True, if at least one param has grad."""
    all_mached = True
    has_acc_bn = False
    for m in model.modules():
        if isinstance(m, MectaNorm2d):
            has_acc_bn = True
            if not m.full_matched:
                all_mached = False
                break
    if has_acc_bn and all_mached:
        return False
    return True
