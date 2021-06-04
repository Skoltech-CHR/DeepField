"""Common custom blocks."""
from torch import nn


def nonlinearity(kind='PReLU', **kwargs):
    """Nonlinearity to use (default to PReLU)."""
    return getattr(nn, kind)(**kwargs)


def norm(dim):
    """Default normalization layer to use (GroupNorm)."""
    return nn.GroupNorm(min(32, dim), dim)
