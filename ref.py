'From https://github.com/Dao-AILab/fast-hadamard-transform/blob/master/fast_hadamard_transform/fast_hadamard_transform_interface.py'

from scipy.linalg import hadamard
import math
import torch
from torch.nn import functional as F

def hadamard_transform_ref(x, scale=1.0):
    """
    x: (..., dim), padded to the next power of 2
    out: (..., dim)
    """
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2 ** log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out = F.linear(x, torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device))
    out = out * scale
    return out[..., :dim].reshape(*x_shape)