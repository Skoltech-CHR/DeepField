"""Miscellaneous utils."""
import numpy as np
import torch
from torch import nn


class LinearInterpolator(nn.Module):
    """Piecewise linear interpolator."""
    def __init__(self, points, values, at_bounds='linear'):
        super().__init__()
        if points.shape[0] != values.shape[0]:
            raise ValueError('Number of points and values should be equal!')
        if points.shape[0] == 1:
            raise ValueError('Minimum of two points is required.')
        if points.ndim > 1:
            raise ValueError('Interpolation is possible only for 1-d points.')
        self.points, sorted_indices = torch.sort(points)
        self.values = values[sorted_indices]
        self.at_bounds = at_bounds

    def __call__(self, x):
        points, values = self._add_bounds(x)
        i0 = find_best_match_indices(x.squeeze(), points.squeeze(), less_or_equal=True)
        i1 = find_best_match_indices(x.squeeze(), points.squeeze(), greater_or_equal=True)
        x0, y0 = points[i0], values[i0]
        x1, y1 = points[i1], values[i1]
        return self._linear_interpolate(x, x0, x1, y0, y1)

    def _add_bounds(self, x):
        max_x = torch.max(x)
        max_x = max_x if max_x > self.points[-1] else None
        min_x = torch.min(x)
        min_x = min_x if min_x < self.points[0] else None

        if self.at_bounds in ('linear', 'constant'):
            return self._add_linear_or_constant_bounds(min_x, max_x)
        if self.at_bounds == 'error':
            raise ValueError('Some of the points to be interpolated are outside of the interpolation set.')
        raise ValueError('Unknown behavior at bounds specified: "%s"' % self.at_bounds)

    def _add_linear_or_constant_bounds(self, min_x, max_x):
        if min_x is None and max_x is None:
            return self.points, self.values
        new_points, new_values = [], []
        if min_x is not None:
            new_points.append(min_x)
            if self.at_bounds == 'constant':
                new_values.append(self.values[0])
            else:
                new_values.append(
                    self._linear_interpolate(min_x, self.points[0], self.points[1], self.values[0], self.values[1])
                )
        if max_x is not None:
            new_points.append(max_x)
            if self.at_bounds == 'constant':
                new_values.append(self.values[-1])
            else:
                new_values.append(
                    self._linear_interpolate(max_x, self.points[-2], self.points[-1], self.values[-2], self.values[-1])
                )
        new_points = [i.unsqueeze(0) for i in new_points]
        new_values = [i.unsqueeze(0) for i in new_values]
        return torch.cat([self.points] + new_points), torch.cat([self.values] + new_values)

    @staticmethod
    def _linear_interpolate(x, x0, x1, y0, y1):
        delta = x1 - x0
        is_degenerate = delta == 0
        not_degenerate = torch.logical_not(is_degenerate)
        delta = delta.clone() * not_degenerate + is_degenerate
        non_degenerate_y = y0 + ((y1 - y0).T * (x - x0) / delta).T
        y = (y0.T * is_degenerate + non_degenerate_y.T * not_degenerate).T
        return y


def get_model_device(model):
    """Get device on which model in situated.

    Parameters
    ----------
    model: nn.Module

    Returns
    -------
    device: torch.device
    """
    return model.parameters().__next__().device


def get_number_of_params(module):
    """Calculate number of trainable parameters in the module.

    Parameters
    ----------
    module: nn.Module

    Returns
    -------
    n: int
        Number of parameters
    """
    parameters = filter(lambda p: p.requires_grad, module.parameters())
    return np.sum([np.prod(p.size()) for p in parameters], dtype=np.int)


def lr_lambda(epoch, base=0.99, exponent=0.05):
    """Multiplier used for learning rate scheduling.

    Parameters
    ----------
    epoch: int
    base: float
    exponent: float

    Returns
    -------
    multiplier: float
    """
    return base ** (exponent * epoch)


def find_best_match_indices(search_for, search_in, less_or_equal=False, greater_or_equal=False):
    """For each element of `search_for`, find the index of closest element in `search_in`.

    Parameters
    ----------
    search_for: torch.tensor
    search_in: torch.tensor
    less_or_equal: bool
        If True, searches for an element which is less or equal than the reference from `search_for`
    greater_or_equal: bool
        If True, searches for an element which is greater or equal than the reference from `search_for`

    Returns
    -------
    indices: torch.Tensor
        Shape: `search_for.shape`
        If the required neighbour was not found, -1 value is used instead of true indices.
    """
    assert search_for.ndim <= 1, "search_for should be a scalar or 1D tensor."
    assert search_in.ndim == 1, "search_in should be a 1D tensor."
    diff = search_for.float().view(-1, 1) - search_in.float().view(1, -1)
    if less_or_equal:
        diff[diff < 0] = torch.tensor(float('inf')).to(search_in.device)  # pylint: disable=not-callable
    if greater_or_equal:
        diff[diff > 0] = torch.tensor(float('inf')).to(search_in.device)  # pylint: disable=not-callable
    diff = torch.abs(diff)
    res = torch.argmin(diff, dim=1)
    if less_or_equal or greater_or_equal:
        res[torch.all(diff == float('inf'), dim=1)] = -1
    return res if search_for.ndim else res.squeeze()
