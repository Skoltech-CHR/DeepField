"""Losses of prediction."""
from torch.nn import MSELoss


class BaseLoss:
    """Base class for calculation of (possibly masked) losses."""
    def __call__(self, ref, pred, mask=None, **kwargs):
        """Calculate loss between reference and prediction."""
        raise NotImplementedError('Abstract method is not implemented.')


class L2Loss(BaseLoss):
    """Class for L2 loss calculation."""
    def __init__(self):
        self._point_wise_l2 = MSELoss(reduction='none')

    def __call__(self, ref, pred, mask=None, **kwargs):
        if mask is None:
            return self._point_wise_l2(pred, ref).mean()
        return self._point_wise_l2(pred, ref).transpose(0, -mask.ndim)[..., mask == 1].mean()


def standardize_loss_pattern(loss_pattern):
    """Transforms given `loss_pattern` to a standard form: tuple of dicts.

    Can infer loss patterns from shortcuts:
        None     ->  dict(mask=None, multiplier=1, loss_fn=L2Loss())
        BaseLoss ->  dict(mask=None, multiplier=1, loss_fn=BaseLoss())
        float    ->  dict(mask=None, multiplier=float, loss_fn=L2Loss())
        str      ->  dict(mask=str, multiplier=1, loss_fn=L2Loss())

    if loss_pattern not isinstance(tuple):
        # will wrap with tuple
    else:
        # will standardize all the items

    Parameters
    ----------
    loss_pattern

    Returns
    -------
    loss_pattern: tuple of dicts
    """
    default = dict(mask=None, multiplier=1, loss_fn=L2Loss())
    types = dict(mask=str, multiplier=(int, float), loss_fn=BaseLoss)
    if isinstance(loss_pattern, (tuple, list)):
        return tuple(standardize_loss_pattern(pattern)[0] for pattern in loss_pattern)
    if loss_pattern is None:
        return tuple([default])
    if isinstance(loss_pattern, dict):
        for key, t in types.items():
            if key not in loss_pattern:
                loss_pattern[key] = default[key]
        return tuple([loss_pattern])
    for key, t in types.items():
        if isinstance(loss_pattern, t):
            return tuple([{k: (v if k != key else loss_pattern) for k, v in default.items()}])
    raise ValueError('Unknown loss_pattern found!')
