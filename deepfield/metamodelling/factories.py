"""Tools for fast custom module creation."""
import numpy as np
from torch import nn

from .custom_blocks import norm, nonlinearity, ResidualBlock, MultiInputSequential, TimeInvariantWrapper


def sequential_factory(n, conv_module, in_ch, ch, *args, use_norm=True, use_nonlin=True, residual_connections=(),
                       wrappers=(MultiInputSequential, TimeInvariantWrapper), **kwargs):
    """Helper for creation of sequential modules. Module is constructed based on the chosen `conv_module`.

    Parameters
    ----------
    n: int
        Number of conv_modules to stack.
    conv_module: nn.Module
        Base module from which sequential is constructed.
    in_ch: int
        Number of input channels in the resulting sequential.
    ch: int, list
        Number of output channels in each of `conv_module`s.
        If `int` is passed, will use similar number of output channels across the modules.
        If `list` is passed, use it's entries. List should have length n.
    args: tuple
        Any additional args passed to the constructor of `conv_module'
        Each arg should be either list or any. If arg is not a list, use similar values across modules.
    use_norm: bool, nn.Module
        If bool, marks the need of using normalization between modules (before the nonlin).
        If nn.Module, uses this module as a normalization.
        Default: True
    use_nonlin: bool, nn.Module
        If bool, marks the need of using nonlinearity between modules.
        If nn.Module, uses this module as a nonlinearity.
        Default: True
    residual_connections: tuple
        Tuple of pairs. Each pair marks which layers should be connected.
        If pair (i, j) is given, connects the input of i-th layer and the output of j-th layer.
        Pairs should not cross each other.
        Default: ()
    wrappers: tuple, list
        List of wrappers, which will be used after constructing the composing modules. Order matters.
        Default: (MultiInputSequential, TimeInvariantWrapper)
    kwargs: dict
        Any additional named args passed to the constructor of `conv_module'
        Each kwarg value should be either list or any. If value is not a list, use similar values across modules.

    Returns
    -------
    out: list, nn.Module
        Composed modules, possibly with applied norm, nonlin, wrappers.
    """

    ch = [in_ch] + _to_list(ch, n)
    args = tuple(_to_list(arg, n) for arg in args)
    kwargs = {key: _to_list(arg, n) for key, arg in kwargs.items()}

    layers = _compose_layers(conv_module, ch, *args, use_norm=use_norm, use_nonlin=use_nonlin, **kwargs)
    layers = _wrap_residuals(layers, residual_connections, bool(use_norm), bool(use_nonlin))
    return _wrap(layers, wrappers)


def _to_list(x, n):
    """Converts x into list by repeating it n times.
    If x is already a list and it has length n, returns x.
    Else, if x is a list and has different length, raises ValueError."""
    if isinstance(x, list):
        if len(x) != n:
            raise ValueError(
                '''If list is passed, it should have len(x) == n.
                Found: n = %d, len(x) = %d''' % (n, len(x))
            )
        return x
    return [x] * n


def _wrap(module, wrappers):
    """Wraps the module with given wrappers. Order is important."""
    for wrapper in wrappers:
        if wrapper in (nn.Sequential, MultiInputSequential):
            module = wrapper(*module)
        else:
            module = wrapper(module)
    return module


def _wrap_residuals(layers, residual_connections, use_norm, use_nonlin):
    """Connects chosen layers with skip connections.

    Parameters
    ----------
    layers: list
        List of layers.
    residual_connections: tuple
        Tuple of pairs. Each pair marks which layers should be connected.
        If pair (i, j) is given, connects the input of i-th layer and the output of j-th layer.
        Pairs should not cross each other.
    use_norm: bool
    use_nonlin: bool

    Returns
    -------
    wrapped_layers: list
    """
    n_other = int(use_norm) + int(use_nonlin)

    residual_connections = np.array(residual_connections)
    # Reindex modules with regard to use_norm and use_nonlin
    residual_connections += residual_connections * n_other

    while residual_connections.shape[0]:
        link, residual_connections = _pop_innermost_residual(residual_connections)

        res_block = ResidualBlock(
            MultiInputSequential(*layers[link[0]: link[1] + 1])
        )
        layers = layers[:link[0]] + [res_block] + layers[link[1] + 1 + n_other:]

        # Reindex modules with respect to changed block
        residual_connections[link[1] + 1:] -= link[1] - link[0] + n_other
    return layers


def _pop_innermost_residual(residual_connections):
    """Pops one residual pair from a list of pairs. If pair are embeded, return innermost one, else the last one.

    Parameters
    ----------
    residual_connections: np.array
        List of residual pairs in the array-like form.
        Shape: [n_pairs, 2]

    Returns
    -------
    poped_pair: np.array
        Shape: [2, ]
    remaining_pairs: np.array
        Shape: [n_pairs-1, 2]
    """
    start_ind = residual_connections[:, 0] == np.max(residual_connections[:, 0])
    end_ind = residual_connections[:, 1] == np.min(residual_connections[start_ind, 1])
    link_ind = np.nonzero(np.all([start_ind, end_ind], axis=0))[0][0]
    link = residual_connections[link_ind]
    residual_connections = np.vstack([residual_connections[:link_ind], residual_connections[link_ind + 1:]])
    return link, residual_connections


def _compose_layers(conv_module, ch, *args, use_norm=True, use_nonlin=True, **kwargs):
    """Compose list of layers, based on ch and additional args and kwargs."""
    norm_layer = norm if isinstance(use_norm, bool) else use_norm
    nonlin_layer = nonlinearity if isinstance(use_nonlin, bool) else use_nonlin

    in_ch = ch.pop(0)
    layers = []
    while ch:
        if layers:
            if use_norm:
                layers.append(norm_layer(in_ch))
            if use_nonlin:
                layers.append(nonlin_layer())

        layer_args = tuple(arg[0] for arg in args)
        args = tuple(arg[1:] for arg in args)

        layer_kwargs = {key: arg[0] for key, arg in kwargs.items()}
        kwargs = {key: arg[1:] for key, arg in kwargs.items()}

        layers.append(conv_module(in_ch, ch[0], *layer_args, **layer_kwargs))
        in_ch = ch.pop(0)
    return layers
