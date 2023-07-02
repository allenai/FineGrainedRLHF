# from: https://github.com/liujch1998/rainier/

import json
from pathlib import Path
from typing import TypeVar, Iterable, List, Union, Any
import random
import numpy as np
import torch
from tqdm.auto import tqdm
import os
import collections
import re

NEGATIVE_INF = -100000.0

T = TypeVar('T')

def reduce_sum(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask)
    return torch.sum(value * mask, axis)


def reduce_mean(value, mask, axis=None):
    if axis is None:
        return torch.sum(value * mask) / torch.sum(mask)
    return reduce_sum(value, mask, axis) / torch.sum(mask, axis)


def reduce_std(value, mask):
    return torch.sqrt(reduce_mean(torch.square(value), mask) - torch.square(reduce_mean(value, mask)))


def reduce_var(value, mask):
    return reduce_mean(torch.square(value), mask) - torch.square(reduce_mean(value, mask))


def logits_to_entropy(logits):
    distribution = torch.distributions.Categorical(logits=logits)
    return distribution.entropy()


def mask_pad(value, mask, pad_value=None):
    if pad_value is None:
        pad_value = NEGATIVE_INF
    return value * mask + pad_value * (1 - mask)


def clamp(value, min_value, max_value):
    return torch.max(torch.min(value, max_value), min_value)


def ceil_div(a, b):
    return (a - 1) // b + 1


def exact_div(a, b):
    q = a // b
    if a != q * b:
        raise ValueError('Inexact division: %s / %s = %s' % (a, b, a / b))
    return q


def whiten(values, masks, shift_mean=True, accelerator=None):
    if accelerator is not None:
        all_values = accelerator.gather(values) # (num_gpus * B, KL)
        all_masks = accelerator.gather(masks) # (num_gpus * B, KL)
        mean, var = reduce_mean(all_values, all_masks), reduce_std(all_values, all_masks)
    else:
        mean, var = reduce_mean(values, masks), reduce_std(values, masks)
    # if accelerator is not None and accelerator.is_main_process:
    #     print(f'all_values: {all_values}, all_masks: {all_masks}')
    #     print(f'mean: {mean}, var: {var}')
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def flatten_dict(nested, sep='.'):
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v
    flat = {}
    rec(nested, '', flat)
    return flat


def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def batchify(data: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    assert batch_size > 0

    batch = []
    for item in data:
        # Yield next batch
        if len(batch) == batch_size:
            yield batch
            batch = []

        batch.append(item)

    # Yield last un-filled batch
    if len(batch) != 0:
        yield batch


def set_seed(seed=19260817, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available() and cuda_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
    with open(file) as f:
        for line in f:
            yield json.loads(line)


def load_cache(file: Path):
    if file.exists():
        with file.open() as f:
            for line in tqdm(f, desc=f'Loading cache from {file}'):
                yield json.loads(line)


def args_to_filename(args):
    return f'_reward-{args.reward_shape}'
    '''
    return "_klCoef" + str(args.kl_coef) + \
        "_lr" + str(args.lr) + \
        "_batchSize" + str(args.batch_size) + \
        "_eps" + str(args.total_episodes) + \
        "_temp" + str(args.temperature) + \
        "_initModel_" + str(args.init_model_type) + \
        "_refModel_" + str(args.ref_model_type) + \
        "_valModel_" + str(args.value_model_type) + \
        "_respLen" + str(args.response_length) + \
        "_realKL_" + str(args.real_kl)
    '''

def get_tensorboard_logname(comment=""):
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join(
        'runs', current_time + '_' + socket.gethostname() + comment)
    return log_dir
