from enum import Enum
import torch
from torch import nn
from robustbench.utils import load_model
from robustbench.model_zoo.enums import ThreatModel, BenchmarkDataset
import warnings

from utils.utils import unwrap_module
from utils.config import MODEL_PATHS

from .batch_norm import MectaNorm2d
from .gc_model import make_checkpoint_resnet


ALL_CIFAR10_MODELS = ['resnet18', 'rb_resnet18', 'rb_wrn18_2', 'rb_ResNeXt29_32x4d', 'rb_wrn28_10_std']
ALL_CIFAR100_MODELS = ['rb_wrn18_2', 'rb_resnet18', 'rb_ResNeXt29_32x4d']
ALL_IN_MODELS = ['resnet18', 'rb_resnet50', 'MobNetV2', 'MobNetV3', 'VGG16BN', 'VGG19BN', 'WRN50x2', 'WRN101x2']


def prepare_model(args, record_bn_cache=False):
    if args.data == 'cifar10':
        if args.model.startswith('rb_'):
            raw_model_name = args.model[len('rb_'):]
            if raw_model_name == 'ResNeXt29_32x4d':
                model_name = 'Hendrycks2020AugMix_ResNeXt'
            elif raw_model_name == 'resnet18':
                model_name = "Kireev2021Effectiveness_RLATAugMix"
            elif raw_model_name == 'wrn28_10_std':
                model_name = 'Standard'
            else:
                raise NotImplementedError(f"model: {args.model}")
            subnet = load_model(model_name=model_name,
                                dataset=BenchmarkDataset.cifar_10,
                                threat_model=ThreatModel.corruptions,
                                model_dir=MODEL_PATHS['RobustBench_root'])
            subnet = make_checkpoint_resnet(
                subnet, args.layer_grad_chkpt_segment, record_bn_cache=record_bn_cache)
    elif args.data in ['IN']:
        if args.alg == 'arm':
            raise NotImplementedError(f"To load models")
        else:
            # List of IN pytorch pre-trained models: https://pytorch.org/vision/stable/models.html
            if args.model.startswith('rb_'):
                raw_model_name = args.model[len('rb_'):]
                if raw_model_name == 'resnet50':
                    model_name = "Hendrycks2020Many"
                else:
                    raise NotImplementedError(f"model: {raw_model_name}")
                subnet = load_model(model_name=model_name,
                                    dataset=BenchmarkDataset.imagenet,
                                    threat_model=ThreatModel.corruptions,
                                    model_dir=MODEL_PATHS['RobustBench_root'])
                subnet.model = make_checkpoint_resnet(
                    subnet.model, args.layer_grad_chkpt_segment, record_bn_cache=record_bn_cache)
            elif args.model.lower().startswith('resnet') or args.model.lower().startswith('resnext'):
                import models.eata_resnet as Resnet
                resnet_kwargs = dict(
                    pretrained=True, layer_grad_chkpt_segment=args.layer_grad_chkpt_segment)
                subnet = Resnet.__dict__[args.model](**resnet_kwargs)
                subnet = make_checkpoint_resnet(
                    subnet, args.layer_grad_chkpt_segment, record_bn_cache=record_bn_cache)
            
            elif args.model == 'DenseNet121':    
                from torchvision.models.densenet import densenet121, DenseNet121_Weights
                subnet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
            elif args.model == 'DenseNet201':
                from torchvision.models.densenet import densenet201, DenseNet201_Weights
                subnet = densenet201(weights=DenseNet201_Weights.IMAGENET1K_V1)
            elif args.model == 'EffNetV2S':
                from torchvision.models.efficientnet import efficientnet_v2_s, EfficientNet_V2_S_Weights
                subnet = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
            elif args.model == 'EffNetV2M':
                from torchvision.models.efficientnet import efficientnet_v2_m, EfficientNet_V2_M_Weights
                subnet = efficientnet_v2_m(weights=EfficientNet_V2_M_Weights.IMAGENET1K_V1)
            elif args.model == 'EffNetV2L':
                from torchvision.models.efficientnet import efficientnet_v2_l, EfficientNet_V2_L_Weights
                subnet = efficientnet_v2_l(weights=EfficientNet_V2_L_Weights.IMAGENET1K_V1)
            elif args.model == 'MobNetV2':
                from torchvision.models.mobilenetv2 import mobilenet_v2, MobileNet_V2_Weights
                subnet = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
            elif args.model == 'MobNetV3':
                from torchvision.models.mobilenetv3 import mobilenet_v3_large, MobileNet_V3_Large_Weights
                subnet = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V2)
            elif args.model == 'VGG16BN':
                from torchvision.models.vgg import vgg16_bn, VGG16_BN_Weights
                subnet = vgg16_bn(weights=VGG16_BN_Weights.IMAGENET1K_V1)
            elif args.model == 'VGG19BN':
                from torchvision.models.vgg import vgg19_bn, VGG19_BN_Weights
                subnet = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)
            elif args.model == 'WRN50x2':
                from torchvision.models.resnet import wide_resnet50_2, Wide_ResNet50_2_Weights
                subnet = wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
            elif args.model == 'WRN101x2':
                from torchvision.models.resnet import wide_resnet101_2, Wide_ResNet101_2_Weights
                subnet = wide_resnet101_2(weights=Wide_ResNet101_2_Weights.IMAGENET1K_V2)
            elif args.model == 'ViT':
                from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
                subnet = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            else:
                raise NotImplementedError(f"model: {args.model} for data {args.data}")

    elif args.data == 'cifar100':

        if args.model == 'rb_resnet18':
            model_name = "Modas2021PRIMEResNet18"
        elif args.model == 'rb_ResNeXt29_32x4d':
            model_name = "Hendrycks2020AugMix_ResNeXt"
        else:
            raise NotImplementedError(f"model: {args.model}")
        subnet = load_model(model_name=model_name, dataset='cifar100',
                            threat_model=ThreatModel.corruptions,
                            model_dir=MODEL_PATHS['RobustBench_root'])
        subnet = make_checkpoint_resnet(
            subnet, args.layer_grad_chkpt_segment, record_bn_cache=record_bn_cache)
    else:
        raise NotImplementedError()

    if args.accum_bn:
        n_repalced = replace_bn(subnet, args.model,
                   use_forget_gate=args.forget_gate,
                   init_beta=args.init_beta, beta=args.beta,
                   dist_metric=args.bn_dist_metric,
                   bn_dist_scale=args.bn_dist_scale,
                   beta_thre=args.beta_thre,
                   prune_q=args.prune_q,
                   )
        n_bn = count_bn(subnet)
        if n_repalced != n_bn:
            warnings.warn(f"Replaced {n_repalced} BNs but actually have {n_bn}. Need to update `replace_bn`.")

        m_cnt = 0
        for m in subnet.modules():
            if isinstance(m, MectaNorm2d):
                # m.update_accum_params(accum_mode='exp', use_forget_gate=args.forget_gate,
                #                       init_beta=args.init_beta, beta=args.beta,
                #                       verbose=m_cnt < 2, dist_metric=args.bn_dist_metric,
                #                       bn_dist_scale=args.bn_dist_scale,
                #                       # var_debias=args.var_debias
                #                       )
                m.reset()
                m_cnt += 1
        assert n_repalced == m_cnt, f"Replaced {n_repalced} BNs but actually inserted {m_cnt} AccumBN."

    return subnet


def replace_bn(model, name, n_repalced=0, **abn_kwargs):
    copy_keys = ['eps', 'momentum', 'affine', 'track_running_stats']

    for mod_name, target_mod in model.named_children():
        # print(f"## inspect module: {name}.{mod_name}")
        if isinstance(target_mod, nn.BatchNorm2d):
            print(f" Insert MECTA-BN to ", name + '.' + mod_name)
            n_repalced += 1

            new_mod = MectaNorm2d(
                target_mod.num_features,
                **{k: getattr(target_mod, k) for k in copy_keys},
                **abn_kwargs,
                name=f'{name}.{mod_name}'
            )
            new_mod.load_state_dict(target_mod.state_dict())
            setattr(model, mod_name, new_mod)
        else:
            n_repalced = replace_bn(
                target_mod, name + '.' + mod_name, n_repalced=n_repalced, **abn_kwargs)
    return n_repalced

def count_bn(model: nn.Module):
    cnt = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            cnt += 1
    return cnt


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='IN', choices=['cifar10', 'IN', 'IN10', 'TIN', 'cifar100'],
                        help='dataset', type=str)
    
    args = parser.parse_args()

    args.alg = 'tent'
    args.accum_bn = True
    args.init_beta = None
    args.beta = 0.1
    args.forget_gate = False
    args.bn_dist_metric = 'simple'
    args.bn_dist_scale = 1.
    args.prune_q = 0
    args.beta_thre = 0.

    model_list = {'IN': ALL_IN_MODELS, 'cifar10': ALL_CIFAR10_MODELS, 'cifar100': ALL_CIFAR100_MODELS}

    for model_name in model_list[args.data]:
        args.model = model_name
        print(f"\n======== {model_name} =======")
        model = prepare_model(args)
    # m = nn.Sequential(nn.BatchNorm2d(10))
    # n_replaced = replace_bn(model, '')
    # n_bn = count_bn(model)
    # print(f'n_replaced: {n_replaced}, #bn={n_bn}')
