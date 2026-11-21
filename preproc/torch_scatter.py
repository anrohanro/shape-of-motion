"""Minimal pure-PyTorch stand-in for the `torch_scatter` package.

DROID-SLAM only needs scatter_sum / scatter_mean with a 1-D index along one
dim, which torch's index_add_ covers. Used because the real package is not
installed in the `som` env (and we don't want to install into it).
"""
import torch


def scatter_sum(src, index, dim=-1, out=None, dim_size=None):
    dim = dim % src.dim()
    index = index.to(src.device).long().view(-1)
    if dim_size is None:
        dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
    shape = list(src.shape)
    shape[dim] = dim_size
    res = torch.zeros(shape, dtype=src.dtype, device=src.device)
    res.index_add_(dim, index, src)
    if out is not None:
        out.add_(res)
        return out
    return res


def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    dim = dim % src.dim()
    total = scatter_sum(src, index, dim, dim_size=dim_size)
    index = index.to(src.device).long().view(-1)
    count = torch.zeros(total.shape[dim], dtype=src.dtype, device=src.device)
    count.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    view = [1] * total.dim()
    view[dim] = -1
    res = total / count.clamp(min=1).view(view)
    if out is not None:
        out.copy_(res)
        return out
    return res
