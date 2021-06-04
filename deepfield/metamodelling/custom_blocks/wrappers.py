"""Custom wrappers for modules."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class ResidualBlock(nn.Module):
    """Wraps a module with residual connection."""
    def __init__(self, module):
        super().__init__()
        self.wrapped_module = module

    def forward(self, inp, *args, **kwargs):
        """Passes the input through the module wrapped with residual block.

        Parameters
        ----------
        inp: torch.Tensor
        args: tuple, optional
        kwargs dict, optional

        Returns
        -------
        out: torch.Tensor
        """
        return inp + self.wrapped_module(inp, *args, **kwargs)


class MultiInputSequential(nn.Module):
    """Concatenates a sequence of modules as a stack.
    Allows using modules, which can have more than one argument."""
    def __init__(self, *args):
        super().__init__()
        for i, module in enumerate(args):
            self.add_module(str(i), module)

    def forward(self, inp, *args, **kwargs):
        """Passes the input through the stacked block of modules.

        Parameters
        ----------
        inp: torch.Tensor
        args: tuple, optional
        kwargs: dict, optional

        Returns
        -------
        out: torch.Tensor
        """
        _ = kwargs
        for module in self._modules.values():
            try:
                inp = module(inp, *args)
            except TypeError:
                inp = module(inp)
        return inp


class TimeInvariantWrapper(nn.Module):
    """Reshapes an input before and after passing it to the wrapped module.
    Reshape mixes and unmixes time and batch axes. Do nothing if time axis is not presented."""
    def __init__(self, wrapped_module):
        super().__init__()
        self.wrapped_module = wrapped_module
        self._inp_shape = None
        self.channel_n_spatial_dims = 0

    def forward(self, inp, *args, **kwargs):
        """Apply wrapped module to the reshaped input. Reshape mixes batch and time dimensions.
        Outputs a tensor with batch and time dimensions similar to that of input.

        Parameters
        ----------
        inp: torch.tensor
            Shape: [B, T, ...]
        args: tuple, optional
        kwargs: dict, optional

        Returns
        -------
        out: torch.Tensor
            Shape: [B, T, ...]
        """
        inp, args = self._flatten_sequence(inp, *args)
        inp = self.wrapped_module(inp, *args, **kwargs)
        inp = self._unflatten_sequence(inp)
        return inp

    def _flatten_sequence(self, inp, *args):
        if inp.ndim in (5, 6):
            self.channel_n_spatial_dims = 4
        elif inp.ndim in (3, 4):
            self.channel_n_spatial_dims = 2
        else:
            raise ValueError('inp.ndim should be in (3, 4, 5, 6). Found: %s' % inp.ndim)
        self._inp_shape = list(inp.shape)
        inp = inp.view(-1, *self._inp_shape[-self.channel_n_spatial_dims:]).contiguous()
        args = (arg.repeat([inp.shape[0] // arg.shape[0]] + [1] * self.channel_n_spatial_dims) for arg in args)
        return inp, args

    def _unflatten_sequence(self, inp):
        self._inp_shape[-self.channel_n_spatial_dims:] = list(inp.size()[-self.channel_n_spatial_dims:])
        return inp.view(*self._inp_shape)


class CheckpointWrapper(nn.Module):
    """Wraps a module with `checkpoint` from `torch.utils.checkpoint`.
    Also, allows microbatching in order to save memory."""
    def __init__(self, wrapped_module):
        super().__init__()
        self.wrapped_module = wrapped_module
        self._dummy_arg = torch.ones(1, dtype=torch.float32, requires_grad=True)
        self.length_cum = 0

    def forward(self, *args, use_checkpointing=True, max_in_batch=0, preserve_rng_state=True, **kwargs):
        """
        Parameters
        ----------
        args: tuple
            Arguments passed to the forward method of the wrapped module
        use_checkpointing: bool
            If True, will use memory efficient computations. Else, do nothing.
        max_in_batch: int
            If > 0, passes arguments in micro-batches of size max_in_batch
        preserve_rng_state: bool
            See `checkpoint.__doc__`
        kwargs: dict
            Not used.

        Returns
        -------
        out
            The output of the wrapped module's forward method.
        """
        _ = kwargs
        out = None
        for a in self._get_minibatches(*args, length=max_in_batch):
            a = self._checkpointed(*a, use_checkpointing=use_checkpointing, preserve_rng_state=preserve_rng_state)
            if out is None:
                out = torch.ones([args[0].shape[0], *a.shape[1:]]).to(a.device)
            out = self._update_output(a, out)
        return out

    def _checkpointed(self, *args, use_checkpointing=True, **checkpoint_kwargs):
        """Wraps module with `checkpoint` function."""
        if use_checkpointing:
            return checkpoint(self._wrapped_with_dummy_arg, self._dummy_arg, *args, **checkpoint_kwargs)
        return self.wrapped_module(*args)

    def _wrapped_with_dummy_arg(self, dummy_arg, *args):
        """Adds a dummy argument with `requires_grad=True` to the wrapped module."""
        assert dummy_arg == self._dummy_arg, \
            'Dummy arg should never be changed! Target: %s. Found: %s' % (self._dummy_arg, dummy_arg)
        return self.wrapped_module(*args)

    def _get_minibatches(self, *args, length=0):
        """Iterator of micro-batches of args"""
        self.length_cum = 0
        stop = False
        if length > 0:
            while not stop:
                a = []
                for arg in args:
                    a.append(arg[self.length_cum: self.length_cum + length])
                yield tuple(a)
                self.length_cum += a[0].shape[0]
                stop = self.length_cum >= args[0].shape[0]
                if stop:
                    self.length_cum = 0
        else:
            yield args

    def _update_output(self, x, out):
        """Updates full output given new micro-batch."""
        length = x.shape[0]
        out[self.length_cum: self.length_cum + length] = x
        return out
