"""Profile the memory usage."""
import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import wandb

from algorithm.base import AdaptableModule
from models.prepare import prepare_model
from utils.dataset import prepare_imagenet_test_data, prepare_cifar10_test_data, prepare_cifar100_test_data
from utils.utils import set_seed, str2bool
from models.batch_norm import get_last_beta, get_bn_cache_size
from utils.gpu_mem_track import MemTracker

# from torch.profiler import profile, record_function, ProfilerActivity
# from torch.autograd import ProfilerActivity

from utils.count_op import FlopCounterMode
from models.gc_model import get_gc_cache_size
from utils.config import set_torch_hub


def print_mem_info():
    print('-'*40)
    mem_dict = {}
    for metric in ['max_memory_cached', 'max_memory_allocated', 'memory_allocated',
                   'memory_reserved', 'max_memory_reserved' # most close to nvidia-smi
                   ]:
        mem_dict[metric] = eval(f'torch.cuda.{metric}()')
        print(f"{metric:>20s}: {mem_dict[metric] / 1e6:10.2f}MB")
    print('-' * 40)
    return mem_dict

def prepare_alg(args, subnet, prepare_data):
    if args.alg == 'src':
        subnet.eval()
        adapt_model = subnet
    elif args.alg == 'bn':
        subnet.train()
        if not args.accum_bn:
            for m in subnet.modules():
                if isinstance(m, nn.BatchNorm2d):
                    # m.requires_grad_(False)
                    # force use of batch stats in train and eval modes
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None
        adapt_model = subnet
    elif args.alg == 'arm':
        from algorithm.arm import ARM_CML_Adapter
        subnet = ARM_CML_Adapter.configure_model(subnet)
        adapt_model = ARM_CML_Adapter(subnet)
    elif args.alg == 'tent':
        from algorithm.tent import Tent
        subnet = Tent.configure_model(subnet, local_bn=not args.accum_bn, filter=args.n_layer)
        # params, param_names = Tent.collect_params(subnet)
        params, param_names = Tent.collect_params(subnet, filter=args.n_layer)
        optimizer = torch.optim.SGD([{'params': params['affine']}], args.lr,
                                    momentum=args.momentum)
                                    # momentum=0.9)  # TODO adaptive momentum?
        adapt_model = Tent(subnet, optimizer)
    elif args.alg == 'eta':
        from algorithm.eata import EATA
        subnet = EATA.configure_model(subnet, local_bn=not args.accum_bn)
        params, param_names = EATA.collect_params(subnet)
        optimizer = torch.optim.SGD([{'params': params['affine']},],
                                    args.lr, momentum=args.momentum)
        adapt_model = EATA(subnet, optimizer, e_margin=args.e_margin, d_margin=args.d_margin)
    elif args.alg == 'eata':
        from algorithm.eata import EATA, compute_fishers
        subnet = EATA.configure_model(subnet, local_bn=not args.accum_bn, filter=args.n_layer)
        params, param_names = EATA.collect_params(subnet, filter=args.n_layer)
        # params, param_names = EATA.collect_params(subnet)

        # compute fisher info-matrix
        _, fisher_loader = prepare_data(
            'original', 5, args.batch_size,
            subset_size=args.fisher_size, seed=args.seed + 1)
        fishers = compute_fishers(params['affine'], subnet, fisher_loader, args.device)

        optimizer = torch.optim.SGD(params['affine'], args.lr, momentum=args.momentum)
        adapt_model = EATA(subnet, optimizer, fishers, args.fisher_alpha,
                           e_margin=args.e_margin, d_margin=args.d_margin)
    else:
        raise NotImplementedError(f'alg: {args.alg}')
    return adapt_model


def get_args():
    parser = argparse.ArgumentParser(description='CTA memory profile')
    parser.add_argument('--alg', default='eta', choices=['src', 'bn', 'eta', 'eata', 'tent', 'arm',
                                                         't3a'],
                        type=str, help='algorithms: src - source model;  '
                                       'bn - Use mini-batch unless merge_batches=True in group mode.')
    parser.add_argument('--data', default='IN', choices=['cifar10', 'IN', 'IN10', 'TIN', 'cifar100'],
                        help='dataset')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')

    parser.add_argument('--model', default='resnet50', type=str)
    parser.add_argument('--accum_bn', default=False, type=str2bool, help='accumulate BN stats.')
    parser.add_argument('--init_beta', default=None, type=float,
                        help='init beta for accum_bn. Use 1. to avoid using train bn. Default will use the same value as beta.')
    parser.add_argument('--beta', default=0.1, type=float, help='beta for accum_bn.')
    parser.add_argument('--forget_gate', default=False, type=str2bool, help='use forget gate.')
    parser.add_argument('--bn_dist_metric', default='simple', type=str,
                        choices=['kl', 'skl', 'skl2', 'simple', 'mmd'])
    parser.add_argument('--bn_dist_scale', default=1., type=float)
    parser.add_argument('--prune_q', default=0., type=float, help='q is the rate of parameters to remove. If is zero, all parameters will be kept.')
    parser.add_argument('--beta_thre', default=0., type=float, help='minimal threshold for beta to do caching. If is zero, all layers will cache.')
    parser.add_argument('--lr', default=0.00025, type=float, help='learning rate. Use 1e-4 for IN and 2.5e-4 for cifar')
    parser.add_argument('--momentum', default=0.9, type=float)

    # eata settings
    parser.add_argument('--fisher_clip_by_norm', type=float, default=10.0,
                        help='Clip fisher before it is too large')
    parser.add_argument('--fisher_size', default=2000, type=int,
                        help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000.,
                        help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000) * 0.40,
                        help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05,
                        help='\epsilon in Eqn. (5) for filtering redundant samples')

    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda', type=str, help='device to use.')
    parser.add_argument('--no_log', action='store_true', help='disable logging.')

    parser.add_argument('--enable_grad', default=False, type=str2bool)
    parser.add_argument('--layer_grad_chkpt_segment', default=1, type=int)
    parser.add_argument('--n_layer', default=None, type=int)

    args = parser.parse_args()
    return args


def main(args):
    MemTracker.init_tracker()

    # set random seeds
    if args.seed is not None:
        set_seed(args.seed, True)

    print(args)
    wandb.init(project='MECTA_cta-eval', name=f'{args.data}_{args.model}_{args.alg}',
               config={**vars(args)},
               mode='offline' if args.no_log else 'online')

    # Prepare data
    if args.data == 'cifar10':
        prepare_data = prepare_cifar10_test_data
    elif args.data == 'cifar100':
        prepare_data = prepare_cifar100_test_data
    elif args.data == 'IN':
        prepare_data = prepare_imagenet_test_data
    elif args.data == 'IN10':
        prepare_data = lambda *sargs, **kwargs: prepare_imagenet_test_data(*sargs, **kwargs,
                                                                           num_classes=10)
    else:
        raise NotImplementedError(f"data: {args.data}")

    # Prepare models
    subnet = prepare_model(args, record_bn_cache=True)
    subnet = subnet.to(args.device)

    # print(subnet)

    adapt_model = prepare_alg(args, subnet, prepare_data)

    flop_counter_ctx = FlopCounterMode(adapt_model)

    cache_by_batch_size = []
    bk_cache_by_batch_size = []
    # for batch_size in [8, 16, 32, 64, 128, 256, 512]:
    for batch_size in [8, 16, 32, 64]:
    # for batch_size in [args.batch_size]:
    # for batch_size in [32]:
        print(f"\n\n/////// Run on batch size = {batch_size} ////////\n")
        _, val_loader = prepare_data(
            'gaussian_noise', 5, batch_size, workers=4)
        for i, (images, target) in enumerate(val_loader):
            # print_mem_info()
            images = images.to(args.device)
            target = target.to(args.device)

            MemTracker.track('Before adaptation')

            # with profile(activities=[ProfilerActivity.CPU],
            #              profile_memory=True, record_shapes=True,
            #              with_flops=True) as prof:
            with flop_counter_ctx:
                if not args.enable_grad:
                    with torch.no_grad():
                        output = adapt_model(images)
                else:
                    output = adapt_model(images)
            MemTracker.track('After adaptation')

            max_forward_cs, total_backward_cs, module_cache_sizes = get_bn_cache_size(adapt_model, return_dict=True)
            # if (hasattr(adapt_model, 'model') and adapt_model.model.gc_cache_size is not None) or (hasattr(adapt_model, 'gc_cache_size') and adapt_model.gc_cache_size is not None):

            if args.layer_grad_chkpt_segment > 1:
                gc_cache_size = get_gc_cache_size(adapt_model.model.model 
                                                    if isinstance(adapt_model.model, nn.Sequential) 
                                                    else adapt_model.model)
                print(f"GC cache_size {gc_cache_size / 1e6:.3f} Mb")
            if max_forward_cs is not None and total_backward_cs is not None:
                print(f"BN estimated cache: \n"
                    f" max_forward_cs: {max_forward_cs / 1e6:.1f}Mb\n"
                    f" total_backward_cs: {total_backward_cs / 1e6:.1f}Mb\n"
                    )
                cache_by_batch_size.append(max_forward_cs)
                bk_cache_by_batch_size.append(total_backward_cs)
            print(f" total: {flop_counter_ctx.total_flops/1e9:.6f} GFLOPs")

            mem_dict = print_mem_info()

            wandb.summary.update(mem_dict)

            break
    if len(cache_by_batch_size) > 0:
        print('max_forward_cs', [c / 1e6 for c in cache_by_batch_size])
        print('bk_cache_by_batch_size', [c / 1e6 for c in bk_cache_by_batch_size])
    else:
        print(f"Did not record cache size.")

    wandb.finish()


if __name__ == '__main__':
    set_torch_hub()
    main(get_args())

