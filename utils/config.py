"""Configuration file for defining paths to data."""
import os

def make_if_not_exist(p):
    if not os.path.exists(p):
        os.makedirs(p)

hostname = os.uname()[1]  # type: str
# Update your paths here.
CHECKPOINT_ROOT = './checkpoint'
if hostname.startswith('illidan') and int(hostname.split('-')[-1]) >= 8:
    data_root = '/localscratch2/jyhong/'
elif hostname.startswith('illidan'):
    data_root = '/media/Research/jyhong/data'
elif hostname.startswith('ip-'):
    data_root = os.path.expanduser('~/data')
else:
    data_root = './data'
hub_root = os.path.join(data_root, 'cache/torch/hub')  # for torch.hub
make_if_not_exist(data_root)
make_if_not_exist(CHECKPOINT_ROOT)
make_if_not_exist(hub_root)

DATA_PATHS = {}

DATA_PATHS = {
    "Cifar10": data_root + "/cifar10",
    "Cifar100": data_root + "/cifar100",
    "CIFAR-10_root": data_root + "/cifar10/",
    "IN": data_root + "/image-net-all/ILSVRC2012/",
    "IN-C": data_root + "/imagenet-c/",
}
MODEL_PATHS = {
    'RobustBench_root': data_root + "/robustbench_models",
}
cifar10_pretrained_fp = f'repos/pytorch-cifar/checkpoint/resnet18_lr0.1.pth'


def set_torch_hub():
    import torch
    torch.hub.set_dir(hub_root)
