"""Utils for evaluation"""
import sys
import time
import torch
import wandb
import numpy as np
from tqdm import tqdm
from algorithm.base import AdaptableModule
from utils.cli_utils import AverageMeter, ProgressMeter, accuracy
from typing import List
from models.batch_norm import get_last_beta, get_bn_cache_size


def validate(val_loader, model, device, stop_at_step=-1):
    batch_time = AverageMeter('Time', ':6.3f')
    acc_mt = AverageMeter('Acc', ':6.2f')
    beta_mt = AverageMeter('Beta', ':6.3f')
    beta_std_mt = AverageMeter('Beta std', ':6.3f')
    cache_mt = AverageMeter('Cache', ':6.3f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, acc_mt, beta_mt, beta_std_mt, cache_mt],
        prefix='Test: ')

    with torch.no_grad():
        for i, dl in enumerate(val_loader):
            end = time.time()
            images, target = dl[0], dl[1]
            images = images.to(device)
            target = target.to(device)
            output = model(images)
            # measure accuracy and record loss
            acc1 = accuracy(output, target, topk=(1,))[0]
            acc_mt.update(acc1, images.size(0))
            # top5.update(acc5[0], images.size(0))

            # measure elapsed time
            cur_batch_time = time.time() - end
            batch_time.update(cur_batch_time)
            # end = time.time()

            # get AccumBN beta
            betas = get_last_beta(model)
            if len(betas) > 0:
                beta_mt.update(np.mean(betas))
                beta_std_mt.update(np.std(betas))
                wandb.log({f'beta/layer-{ib}': b for ib, b in enumerate(betas)}, commit=False)
                wandb.log({f'mean beta': np.mean(betas)}, commit=False)

            max_forward_cs, backward_cs = get_bn_cache_size(model)
            if backward_cs is not None and max_forward_cs is not None:
                cache_size = max([max_forward_cs, backward_cs])
                cache_mt.update(cache_size / 1e6)
                wandb.log({'cache size (MB)': cache_size / 1e6,}, commit=True)

            if i % 50 == 0:
                progress.display(i)
            wandb.log({'batch acc': acc1}, commit=True)

            if stop_at_step > 0 and i >= stop_at_step:
                break
    return acc_mt.avg, cache_mt.max, cache_mt.avg


def group_validate(val_loader, adapt_loaders: List, model, device, adapt_batches=None, n_batch=1,
                   merge_batches=False, stop_at_step=-1,
                   display_interval=50):
    """Validate model using a group of data/batches.

    Args:
        adapt_loaders (list): Loader for adaptation whose acc will not be counted.
        val_loader: Loader for validation whose acc is counted.
        adapt_batches (list): Batch sizes for every adapt_loader. If not provided, equals n_batch-1.
        n_batch (int): Number of batches for adaptation, where we reset model after n batches.
    """
    assert isinstance(adapt_loaders, list)
    batch_time = AverageMeter('Time', ':6.3f')
    acc_mt = AverageMeter('Acc', ':6.2f')
    pair_sup_acc_mt = AverageMeter('Sup-Acc', ':6.2f')
    # top5 = AverageMeter('Acc@5', ':6.2f')
    beta_mt = AverageMeter('Beta', ':6.2f')
    beta_std_mt = AverageMeter('Beta std', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, acc_mt, pair_sup_acc_mt, beta_mt, beta_std_mt],
        prefix='Test: ')

    # n_adapt_batch = n_batch - 1  # adaptation-only batches
    if adapt_batches == None:
        assert len(adapt_loaders) == 1
        adapt_batches = [n_batch-1]
    else:
        assert sum(adapt_batches) == n_batch - 1

    with torch.no_grad():
        end = time.time()

        for i, (images, target) in enumerate(tqdm(val_loader)):
            support_acc_mt = AverageMeter('Acc', ':6.2f')
            if merge_batches:
                batch_list = []
            for j, adapt_loader in enumerate(adapt_loaders):
                ada_iter = iter(adapt_loader)
                n_adapt_batch = adapt_batches[j]
                for _ in range(n_adapt_batch):
                    try:
                        ada_imgs, ada_trgs = next(ada_iter)
                    except StopIteration:
                        ada_iter = iter(adapt_loader)
                        ada_imgs, ada_trgs = next(ada_iter)
                    ada_imgs, ada_trgs = ada_imgs.to(device), ada_trgs.to(device)
                    if merge_batches:
                        batch_list.append(ada_imgs)
                    else:
                        ada_output = model(ada_imgs)

                        ada_acc = accuracy(ada_output, ada_trgs, topk=(1,))[0]
                        support_acc_mt.update(ada_acc, images.size(0))

            images = images.to(device)
            target = target.to(device)
            if merge_batches:
                pred_len = len(images)
                images = torch.cat(batch_list + [images], dim=0)
                output = model(images)
                output = output[-pred_len:]
            else:
                output = model(images)
            # measure accuracy and record loss
            acc = accuracy(output, target, topk=(1,))[0]
            acc_mt.update(acc, images.size(0))
            pair_sup_acc_mt.update(support_acc_mt.avg, 1)

            # measure elapsed time
            cur_batch_time = time.time() - end
            batch_time.update(cur_batch_time)
            end = time.time()

            # get AccumBN beta
            betas = get_last_beta(model)
            beta_mt.update(np.mean(betas))
            beta_std_mt.update(np.std(betas))

            if i % display_interval == 0:
                progress.display(i, print_fh=lambda s: tqdm.write(s, file=sys.stdout))
            wandb.log({'batch acc': acc}, commit=True)

            if isinstance(model, AdaptableModule):
                # print(f"Reset at batch {i}")
                model.reset_all()

            if stop_at_step > 0 and i >= stop_at_step:
                break
    return acc_mt.avg  # acc_mt.step_avg, acc_mt.step_std, acc_mt.update_cnt
