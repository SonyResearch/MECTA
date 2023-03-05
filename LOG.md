Experiment Logs
===============

## Benchmark on Standard Models (Table 2)

We use the standard ResNet50 pretrained on ImageNet.
```shell
# Baselines
wandb sweep sweeps/continual_IN.yaml
# MECTA
wandb sweep sweeps/continual_IN_mecta_std_batch.yaml
```

## Cache Constrained Benchmarks (Table 1)

```shell
wandb sweep sweeps/continual_IN_mecta.yaml
wandb sweep sweeps/continual_cifar10_mecta.yaml
wandb sweep sweeps/continual_cifar100_mecta.yaml
```

## Ablation study on beta

Ablation on the forget gate with different CTA methods.
```shell
wandb sweep sweeps/continual_IN_mecta_ablate_gate.yaml
# => [03/03] https://wandb.ai/jyhong/MECTA_release/sweeps/7jlh4hm7
#    @GPU9 ablate gate on three methods again.
```

Ablation studies.
* (B) Abalation on the batch size
```shell
wandb sweep sweeps/continual_IN_ablation_batch.yaml
# => [03/03] https://wandb.ai/jyhong/MECTA_release/sweeps/5o2wcgi5
#    @GPU9 ablation on batch size.
```
* (L) Abalation on beta_thre
```shell
wandb sweep sweeps/continual_IN_ablation_beta_thre.yaml
# => [03/03] https://wandb.ai/jyhong/MECTA_release/sweeps/hy3fkd0j
#    @GPU9 ablation on beta_thre
```
* (C) Prune channels
```shell
wandb sweep sweeps/continual_IN_ablation_prune.yaml
# => [03/03] https://wandb.ai/jyhong/MECTA_release/sweeps/fwbdlkoh
#    @GPU9 
```

Pair-eval

