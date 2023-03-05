import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import wandb

from models.prepare import prepare_model
from utils.dataset import prepare_imagenet_test_data, prepare_cifar10_test_data, prepare_cifar100_test_data
from utils.utils import set_seed, str2bool
from utils.eval import validate, group_validate
from utils.config import set_torch_hub


def get_args():
    parser = argparse.ArgumentParser(description='CTA Evaluation')

    # overall experimental settings
    parser.add_argument('--eval_mode', default='continual',
                        choices=['continual', 'pair', 'group'], type=str,
                        help='evaluation mode. \n'
                             'group: Consider a set of continual batches as a test case. Use this'
                             ' to evaluate the adaptation on a fixed number of batches, only '
                             'the last batch (fully adapted in the case) will be counted for Acc.\n'
                             'pair: In a group, we include data from two domains (in different batches).\n'
                             'continual: Like group, but do not reset model after a limited num of batches.')
    parser.add_argument('--alg', default='eta', choices=['src', 'bn', 'eta', 'eata', 'tent', 'arm',
                                                         'cotta', 'cotta_bn'],
                        type=str, help='algorithms: src - source model;  '
                                       'bn - Use mini-batch unless merge_batches=True in group mode.')
    parser.add_argument('--no_log', action='store_true', help='disable logging.')

    # path of data, output dir
    parser.add_argument('--data', default='IN', choices=['cifar10', 'IN', 'IN10', 'TIN', 'cifar100'],
                        help='dataset')
    parser.add_argument('--test_corrupt', default='std', choices=['arm', 'std', 'org'],
                        help='arm - a subset of ARM-used corruptions. std - standard one used in CTA.')
    parser.add_argument('--model', default='resnet50', type=str)

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda', type=str, help='device to use.')
    parser.add_argument('--workers', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--lr', default=0.00025, type=float, help='learning rate. Use 1e-4 for IN and 2.5e-4 for cifar')
    parser.add_argument('--momentum', default=0.9, type=float)

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')

    # batch config for eval
    parser.add_argument('--iters', default=-1, type=int, help='how many iterations for eval. [Default: -1 for all batches]')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--support_batch', default=None, type=int, help='number of batches for support set (default: 1)')
    parser.add_argument('--merge_batches', default=False, type=str2bool,
                        help='whether to merge several batches of images into one batch. '
                             'Effective w/ group eval. Use this to make a large batch from '
                             'a mixture of data domains.')
    parser.add_argument('--cur_batch', default=1, type=int, help='# of current-domain batches used'
                                                                 'in pair evaluation.')

    # MECTA configuration
    parser.add_argument('--accum_bn', default=False, type=str2bool, help='accumulate BN stats.')
    parser.add_argument('--init_beta', default=None, type=float,
                        help='init beta for accum_bn. Use 1. to avoid using train bn. Default will use the same value as beta.')
    parser.add_argument('--beta', default=0.1, type=float, help='beta for accum_bn.')
    parser.add_argument('--forget_gate', default=False, type=str2bool, help='use forget gate.')
    parser.add_argument('--bn_dist_metric', default='skl', type=str,
                        choices=['kl', 'skl', 'skl2', 'simple', 'mmd'])
    parser.add_argument('--bn_dist_scale', default=1., type=float)

    parser.add_argument('--prune_q', default=0., type=float, help='q is the rate of parameters to remove. If is zero, all parameters will be kept.')
    parser.add_argument('--beta_thre', default=0., type=float, help='minimal threshold for beta to do caching. If is zero, all layers will cache.')

    # for ablation study
    parser.add_argument('--n_layer', type=int, default=None, help='For Tent&EATA, num of BN layers to train, start from the output.')
    parser.add_argument('--layer_grad_chkpt_segment', type=int, default=1, help='Num of segments per ResNet stage for gradient checkpointing.')
    args = parser.parse_args()
    if args.eval_mode == 'pair':
        assert args.cur_batch < args.support_batch

    # default args
    # eata settings
    args.fisher_clip_by_norm = 10.
    args.fisher_size = 2000
    if args.data == 'cifar10':
        args.fisher_alpha = 1.
        args.e_margin = math.log(10) * 0.40
        args.d_margin = 0.4
    elif args.data == 'cifar100':
        if args.model in ['rb_ResNeXt29_32x4d']:
            args.fisher_alpha = 2000.
            args.e_margin = math.log(100) * 0.40
            args.d_margin = 0.05
        else:
            raise RuntimeError(f"No pre-set parameters for {args.model} at {args.data}")
    elif args.data == 'IN':
        args.fisher_alpha = 2000.
        args.e_margin = math.log(1000) * 0.40
        args.d_margin = 0.05
    else:
        raise NotImplementedError(f'No default EATA param for data: {args.data}')
    return args


def main(args):
    # set random seeds
    if args.seed is not None:
        set_seed(args.seed, True)

    # all_corruptions = None
    if args.data in ['IN', 'IN10', 'cifar10', 'cifar100', 'TIN']:
        if args.test_corrupt == 'std':  # for continual eval only
            all_corruptions = ['gaussian_noise', 'shot_noise',  'impulse_noise', 'defocus_blur',
                               'glass_blur',     'motion_blur', 'zoom_blur',      'snow',
                               'frost',           'fog',        'brightness',     'contrast',
                               'elastic_transform', 'pixelate', 'jpeg_compression', 'original']
            # excluded noise but provided in cifar10-c: ('saturate', 'spatter', 'gaussian_blur', 'speckle_noise',)
        elif args.test_corrupt == 'arm':
            # NOTE test noise used by ARM. We add some. The set is harder than the standard CTA config.
            all_corruptions = ['impulse_noise', 'motion_blur', 'fog', 'elastic_transform']
        elif args.test_corrupt == 'org':
            all_corruptions = ['original']
        else:
            raise ValueError(f"test_corrupt: {args.test_corrupt}")
    else:
        raise NotImplementedError(f"data: {args.data}")
    print("All corruptions:", all_corruptions)

    print(args)
    wandb.init(project='RobARM_cta-eval', name=f'{args.data}_{args.model}_{args.alg}',
               config={**vars(args), 'all_corruptions': all_corruptions},
               mode='offline' if args.no_log else 'online')

    # Prepare data
    if args.data == 'cifar10':
        prepare_data = prepare_cifar10_test_data
    elif args.data == 'cifar100':
        prepare_data = prepare_cifar100_test_data
    elif args.data == 'IN':
        prepare_data = prepare_imagenet_test_data
    else:
        raise NotImplementedError(f"data: {args.data}")

    # Prepare models
    subnet = prepare_model(args)
    subnet = subnet.to(args.device)

    # Prepare algorithms
    if args.alg == 'src':
        subnet.eval()
        adapt_model = subnet
    elif args.alg == 'bn':
        subnet.train()
        if not args.accum_bn:
            for m in subnet.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(False)
                    # force use of batch stats in train and eval modes
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
        else:
            for m in subnet.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(False)
        adapt_model = subnet
    elif args.alg == 'tent':
        from algorithm.tent import Tent
        subnet = Tent.configure_model(subnet, local_bn=not args.accum_bn, filter=args.n_layer)
        params, param_names = Tent.collect_params(subnet, filter=args.n_layer)
        optimizer = torch.optim.SGD([{'params': params['affine']}], args.lr,
                                    momentum=args.momentum)
        adapt_model = Tent(subnet, optimizer)
    elif args.alg == 'eta':
        from algorithm.eata import EATA
        subnet = EATA.configure_model(
            subnet, local_bn=not args.accum_bn, filter=args.n_layer)
        params, param_names = EATA.collect_params(subnet, filter=args.n_layer)
        optimizer = torch.optim.SGD([{'params': params['affine']},],
                                    args.lr, momentum=args.momentum)
        adapt_model = EATA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin)
    elif args.alg == 'eata':
        from algorithm.eata import EATA, compute_fishers
        subnet = EATA.configure_model(subnet, local_bn=not args.accum_bn, filter=args.n_layer)
        params, param_names = EATA.collect_params(subnet, filter=args.n_layer)

        # compute fisher info-matrix
        _, fisher_loader = prepare_data(
            'original', args.level, args.batch_size, workers=args.workers,
            subset_size=args.fisher_size, seed=args.seed + 1)
        fishers = compute_fishers(params['affine'], subnet, fisher_loader, args.device)

        optimizer = torch.optim.SGD(params['affine'], args.lr, momentum=args.momentum)
        adapt_model = EATA(subnet, optimizer, fishers, args.fisher_alpha,
                           e_margin=args.e_margin, d_margin=args.d_margin)
    elif args.alg in ['cotta', 'cotta_bn']:
        from algorithm.cotta import CoTTA
        assert args.n_layer is None, "Not support partial layer."
        if args.alg == 'cotta_bn':
            subnet = CoTTA.configure_model(
                subnet, bn_only=True, local_bn=not args.accum_bn)
            params, param_names = CoTTA.collect_params(subnet, bn_only=True)
        else:
            subnet = CoTTA.configure_model(
                subnet, bn_only=False, local_bn=not args.accum_bn)
            params, param_names = CoTTA.collect_params(subnet, bn_only=False)

        if args.data == 'cifar10':
            optimizer = torch.optim.Adam(params, lr=args.lr,  # 1e-3 for cifar10
                                        betas=(0.9, 0.999), weight_decay=0.)
            cotta_kwargs = dict(mt_alpha=0.999, rst_m=0.01, ap=0.92)
        elif args.data == 'cifar100':
            optimizer = torch.optim.Adam(params, lr=args.lr,  # 1e-3 for cifar100
                                        betas=(0.9, 0.999), weight_decay=0.)
            cotta_kwargs = dict(mt_alpha=0.999, rst_m=0.01, ap=0.72)
        elif args.data == 'IN':
            optimizer = torch.optim.SGD(params, lr=args.lr,  # 0.01 for IN
                                        momentum=0.9, dampening=0, weight_decay=0., nesterov=True)
            cotta_kwargs = dict()
            from algorithm.cotta import CoTTA_ImageNet as CoTTA
        else:
            raise NotImplementedError(f"data: {args.data}")
        adapt_model = CoTTA(subnet, optimizer, **cotta_kwargs)
    else:
        raise NotImplementedError(f'alg: {args.alg}')

    # Start continual adaptation
    if args.eval_mode == 'continual':
        accs = []
        for i_corrupt, corrupt in enumerate(all_corruptions):
            print('Current corrupt:', corrupt)

            _, val_loader = prepare_data(
                corrupt, args.level, args.batch_size, workers=args.workers)

            acc, max_cache, avg_cache = validate(val_loader, adapt_model, args.device,
                                      stop_at_step=args.iters)
            info = f"[{i_corrupt}] {args.alg}@{corrupt} Acc: {acc:.2f}%"
            if max_cache is not None and avg_cache is not None:
                info += f" Max Cache: {max_cache:.2f} MB, Avg Cache: {avg_cache:.2f} MB"
            print(info)

            if args.alg in ['eata', 'eta']:
                print(
                    f"num of reliable samples is {adapt_model.num_samples_update_1}, "
                    f"num of reliable+non-redundant samples is {adapt_model.num_samples_update_2}")
                wandb.log({'num reliable smp': adapt_model.num_samples_update_1,
                           'num bwd smp': adapt_model.num_samples_update_2},
                          commit=False)
                adapt_model.num_samples_update_1, adapt_model.num_samples_update_2 = 0, 0
            wandb.log({'acc': acc, 'max_cache': max_cache, 'avg_cache': avg_cache, 'corrupt': i_corrupt}, commit=True)
            accs.append(acc)
        wandb.summary['avg acc'] = np.mean(accs)

    elif args.eval_mode == 'group':
        domain_accs = []
        for i_corrupt, corrupt in enumerate(all_corruptions):
            print(f'[{i_corrupt}] {args.alg}@{corrupt}')

            _, adapt_loader = prepare_data(
                corrupt, args.level, args.batch_size, workers=args.workers)
            _, val_loader = prepare_data(
                corrupt, args.level, args.batch_size, workers=args.workers)

            acc = group_validate(
                val_loader, [adapt_loader], adapt_model, args.device, n_batch=args.support_batch,
                merge_batches=args.merge_batches, stop_at_step=args.iters,
            )
            # print(f"DONE - Acc: {acc:.1f}±{acc_std:.1f}% | #Trial: {exe_cnt} ")
            print(f"DONE - Acc: {acc:.1f}% ")
            wandb.log({'acc': acc, 'corrupt': i_corrupt}, commit=True)

            domain_accs.append(acc)
        wandb.summary.update({
            'avg acc': np.mean(domain_accs),
            'worst acc': np.min(domain_accs),
        })

    elif args.eval_mode == 'pair':
        domain_accs = []
        for i_corrupt, corrupt in enumerate(all_corruptions):
            acc_pre = np.zeros(len(all_corruptions)) - 1.
            for i_pre_corrupt, pre_corrupt in enumerate(all_corruptions):
                if i_pre_corrupt == i_corrupt:
                    acc_pre[i_pre_corrupt] = np.nan
                    continue
                print(f'[{i_corrupt}] {args.alg}@{pre_corrupt}>>{corrupt}')

                _, pre_adapt_loader = prepare_data(
                    pre_corrupt, args.level, args.batch_size, workers=args.workers)
                _, adapt_loader = prepare_data(
                    corrupt, args.level, args.batch_size, workers=args.workers)
                _, val_loader = prepare_data(
                    corrupt, args.level, args.batch_size, workers=args.workers)

                acc = group_validate(
                    val_loader, [pre_adapt_loader, adapt_loader], adapt_model, args.device,
                    adapt_batches=[args.support_batch-args.cur_batch, args.cur_batch-1],
                    n_batch=args.support_batch, merge_batches=args.merge_batches,
                    stop_at_step=args.iters,
                )
                # print(f"DONE - Acc: {acc:.1f}±{acc_std:.1f}% | #Trial: {exe_cnt} ")
                print(f"DONE - Acc: {acc:.1f}% ")
                wandb.log({'acc': acc, 'corrupt': i_corrupt}, commit=True)
                acc_pre[i_pre_corrupt] = acc
            avg_acc = np.nanmean(acc_pre)
            worst_acc = np.nanmin(acc_pre)
            worst_corr = np.nanargmin(acc_pre)
            print(f"DONE>{corrupt} - Acc: {avg_acc:.1f}% | Worst Acc {worst_acc:.1}% "
                  f"({worst_corr}: {all_corruptions[worst_corr]})")
            wandb.log({'pair avg acc': avg_acc, 'pair worst acc': worst_acc,
                       'worst corruption': worst_corr}, commit=True)
            domain_accs.append(avg_acc)
        wandb.summary.update({
            'avg acc': np.mean(domain_accs),
            'worst acc': np.min(domain_accs),
        })
    else:
        raise NotImplementedError(f"eval mode: {args.eval_mode}")


if __name__ == '__main__':
    set_torch_hub()
    args = get_args()
    main(args)
