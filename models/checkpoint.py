import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from models.batch_norm import get_bn_cache_size, MectaNorm2d
from utils.utils import dtype_memory_size_dict


class CheckpointSeq(nn.Module):
    def __init__(self, sequential_mod: nn.Sequential, segment=1, record_bn_cache=False) -> None:
        super().__init__()
        assert 1 <= segment <= len(sequential_mod), \
            f"Invalid segment. Should in range [1, {len(sequential_mod)}] but got {segment}"

        self.sequential_mod = sequential_mod
        self.checkpoint_sequential = checkpoint_sequential(record_bn_cache)
        self.segment = segment
        self.cache_size = 0  # TODo need to reset.
        self.max_bn_cache_size = 0
    
    def forward(self, x):
        if self.segment > 1:
            out, cache_size, max_bn_cache_size = self.checkpoint_sequential(self.sequential_mod, self.segment, x)
            self.cache_size = cache_size
            self.max_bn_cache_size = max_bn_cache_size
            return out
        else:
            return self.sequential_mod(x)
    
    def get_split_num(self):
        return len(self.sequential_mod) // self.segment


def print_chkpt_seg(model):
    n_gc = 0
    total_parts = 0
    for nm, mod in model.named_modules():
        if isinstance(mod, CheckpointSeq):
            split = mod.get_split_num()
            total_parts += split
            print(f" Segment stage {nm} into {split} parts.")
            n_gc += 1
    if n_gc > 0:
        print(f"Total segments: {total_parts}")

def compute_input_cache(input):
    # if isinstance(input, tuple):
    #     return sum([torch.prod(torch.tensor(inp.shape)).item() for inp in input if torch.is_tensor(inp)])
    # else:
    return torch.prod(torch.tensor(input.shape)).item()

class checkpoint_sequential:
    def __init__(self, record_bn_cache):
        self.record_bn_cache = record_bn_cache

    def __call__(self, functions, segments, input, **kwargs):
        r"""Modified from torch.utils.checkpoint.checkpoint_sequential

        Returns:
            Output of running :attr:`functions` sequentially on :attr:`*inputs`

        Example:
            >>> model = nn.Sequential(...)
            >>> input_var = checkpoint_sequential(model, chunks, input_var)
        """
        # Hack for keyword-only parameter in a python 2.7-compliant way
        preserve = kwargs.pop('preserve_rng_state', True)
        if kwargs:
            raise ValueError("Unexpected keyword arguments: " +
                            ",".join(arg for arg in kwargs))

        def run_function(start, end, functions):
            def forward(input):
                for j in range(start, end + 1):
                    input = functions[j](input)
                return input
            return forward

        if isinstance(functions, torch.nn.Sequential):
            functions = list(functions.children())

        segment_size = len(functions) // segments
        # the last chunk has to be non-volatile
        end = -1
        cache_size = 0
        max_bn_cache_size = 0
        for start in range(0, segment_size * (segments - 1), segment_size):
            end = start + segment_size - 1
            cache_size += compute_input_cache(input)
            input = checkpoint(run_function(start, end, functions), input,
                            preserve_rng_state=preserve)
            
            if self.record_bn_cache:
                _, bn_cache_size = get_bn_cache_size(
                    torch.nn.Sequential(*[functions[i] for i in range(start, end+1)]))
                if bn_cache_size > max_bn_cache_size:
                    max_bn_cache_size = bn_cache_size
                    # print(f">> bn_cache_size: {bn_cache_size/1e6:.2f} Mb")
        cache_size += compute_input_cache(input)
        start = end + 1
        end = len(functions) - 1
        out = run_function(start, end, functions)(input)
        if self.record_bn_cache:
            _, bn_cache_size = get_bn_cache_size(
                torch.nn.Sequential(*[functions[i] for i in range(start, end+1)]))
            if bn_cache_size > max_bn_cache_size:
                max_bn_cache_size = bn_cache_size
                # print(f">> bn_cache_size: {bn_cache_size/1e6:.2f} Mb")
            
            if max_bn_cache_size == 0 and not input.requires_grad: # This layer does not need cache.
                cache_size = 0.
        else:
            max_bn_cache_size = 0.
        return out, cache_size * dtype_memory_size_dict[torch.float], max_bn_cache_size
