"""Transform models with gradient checkpoint."""
import torch
from torch import nn
import warnings

from torchvision.models import ResNet as ResNet
from robustbench.model_zoo.architectures.resnext import CifarResNeXt

from .batch_norm import MectaNorm2d
from .checkpoint import CheckpointSeq, print_chkpt_seg


def make_checkpoint_resnet(model, layer_segment, record_bn_cache=True):
    if layer_segment > 1:
        if is_resnet(model):
            model = make_checkpoint_resnet_(model, layer_segment, record_bn_cache)
        elif is_cifar_resnetXt(model):
            model = make_checkpoint_resnetXt(model, layer_segment, record_bn_cache)
        else:
            raise NotImplementedError(f"Cannot add grad-checkpoint to {type(model)}")
        print_chkpt_seg(model)
    return model


def get_gc_cache_size(model: ResNet):
    if is_resnet(model):
        return get_gc_cache_size_(model)
    elif is_cifar_resnetXt(model):
        return get_gc_cache_size_resnetXt(model)
    else:
        raise NotImplementedError(f"Cannot get grad-checkpoint cache from {type(model)}")



# ########### ResNet ###########
def is_resnet(model):
    from models.eata_resnet import ResNet as eata_ResNet
    return isinstance(model, (ResNet, eata_ResNet))


def make_checkpoint_resnet_(resnet_model: ResNet, layer_segment, record_bn_cache=True):
    assert is_resnet(resnet_model), f"Require ResNet but got: {resnet_model.__class__.__name__}"
    print(f"Transform {resnet_model.__class__.__name__} with grad-checkpoint {layer_segment}")
    assert isinstance(resnet_model.layer1, nn.Sequential)
    resnet_model.layer1 = CheckpointSeq(resnet_model.layer1, layer_segment, record_bn_cache)

    assert isinstance(resnet_model.layer2, nn.Sequential)
    resnet_model.layer2 = CheckpointSeq(resnet_model.layer2, layer_segment, record_bn_cache)

    assert isinstance(resnet_model.layer3, nn.Sequential)
    resnet_model.layer3 = CheckpointSeq(resnet_model.layer3, layer_segment, record_bn_cache)

    assert isinstance(resnet_model.layer4, nn.Sequential)
    resnet_model.layer4 = CheckpointSeq(resnet_model.layer4, layer_segment, record_bn_cache)
    return resnet_model
    # CifarResNeXt

def get_gc_cache_size_(model: ResNet):
    """
    NOTE: Do not nest CheckpointSeq. Otherwise, the cache size will be wrong.
    NOTE: The model has to be ResNet.
    """
    assert is_resnet(model), f"Require ResNet but got: {model.__class__.__name__}"
    gc_cache_size = 0
    max_bn_cache_size = 0
    n_gc = 0
    for nm, mod in model.named_modules():
        if isinstance(mod, CheckpointSeq):
            gc_cache_size += mod.cache_size
            if mod.max_bn_cache_size > max_bn_cache_size:
                max_bn_cache_size = mod.max_bn_cache_size
            n_gc += 1
    if n_gc == 0:
        warnings.warn(f"Not found gradient checkpoint.")
    else:
        if hasattr(model, 'model'):
            model = model.model
        assert isinstance(model.bn1, MectaNorm2d)
        forward_cache, backward_cache = model.bn1.cache_size()
        gc_cache_size += backward_cache
    return gc_cache_size + max_bn_cache_size


# ########### Cifar ResNetXt ###########
def is_cifar_resnetXt(model):
    return isinstance(model, CifarResNeXt)


def make_checkpoint_resnetXt(resnet_model: CifarResNeXt, layer_segment, record_bn_cache=False):
    assert is_cifar_resnetXt(
        resnet_model), f"Require ResNet but got: {resnet_model.__class__.__name__}"
    print(
        f"Transform {resnet_model.__class__.__name__} with grad-checkpoint {layer_segment}")
    assert isinstance(resnet_model.stage_1, nn.Sequential)
    resnet_model.stage_1 = CheckpointSeq(
        resnet_model.stage_1, layer_segment, record_bn_cache)

    assert isinstance(resnet_model.stage_2, nn.Sequential)
    resnet_model.stage_2 = CheckpointSeq(
        resnet_model.stage_2, layer_segment, record_bn_cache)

    assert isinstance(resnet_model.stage_3, nn.Sequential)
    resnet_model.stage_3 = CheckpointSeq(
        resnet_model.stage_3, layer_segment, record_bn_cache)
    
    return resnet_model


def get_gc_cache_size_resnetXt(model: CifarResNeXt):
    """
    NOTE: Do not nest CheckpointSeq. Otherwise, the cache size will be wrong.
    NOTE: The model has to be ResNet.
    """
    assert is_cifar_resnetXt(
        model), f"Require ResNet but got: {model.__class__.__name__}"
    gc_cache_size = 0
    max_bn_cache_size = 0
    n_gc = 0
    for nm, mod in model.named_modules():
        if isinstance(mod, CheckpointSeq):
            gc_cache_size += mod.cache_size
            if mod.max_bn_cache_size > max_bn_cache_size:
                max_bn_cache_size = mod.max_bn_cache_size
            n_gc += 1
    if n_gc == 0:
        warnings.warn(f"Not found gradient checkpoint.")
    else:
        if hasattr(model, 'model'):
            model = model.model
        assert isinstance(model.bn_1, MectaNorm2d)
        forward_cache, backward_cache = model.bn_1.cache_size()
        gc_cache_size += backward_cache
    return gc_cache_size + max_bn_cache_size
