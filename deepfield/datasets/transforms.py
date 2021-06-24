"""Transforms applied to samples in a FieldDataset."""
from random import choice
import torch
import torch.nn.functional as F
import numpy as np

from ..field.base_component import BaseComponent


NON_NORMALIZED_ATTRS = ['MASKS', 'TABLES', 'GRID']


class change_sample_state:  # pylint: disable=invalid-name
    """Decorator for changing sample's state after the application of the decorated transform."""
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, transform):
        def wrapped(instance, sample, **kwargs):
            sample = transform(instance, sample, **kwargs)
            if hasattr(sample, 'state'):
                for k, v in self.kwargs.items():
                    if hasattr(sample.state, k):
                        sample.set_state(**{k: v})
                    else:
                        sample.init_state(**{k: v})
            return sample
        return wrapped


class Transform:
    """Transform applied to a sample"""
    def __call__(self, sample, inplace=True):
        """Apply a transform."""
        raise NotImplementedError('Abstract method.')

    def __str__(self):
        """Shortened string representation of the class."""
        return super().__str__().split()[0].split('.')[-1]


class ToTensor(Transform):
    """Convert ndarrays in sample to Tensors."""
    @change_sample_state(numpy=False, tensor=True)
    def __call__(self, sample, inplace=True):
        out = sample if inplace else sample.empty_like()
        for comp, value in sample.items():
            if isinstance(value, (dict, BaseComponent)):
                out[comp] = self(value, inplace=inplace)
            else:
                out[comp] = torch.from_numpy(value).float()
        return out


class ToNumpy(Transform):
    """Convert Tensors in sample to ndarrays."""
    @change_sample_state(numpy=True, tensor=False)
    def __call__(self, sample, inplace=True):
        out = sample if inplace else sample.empty_like()
        for comp, value in sample.items():
            if isinstance(value, (dict, BaseComponent)):
                out[comp] = self(value, inplace=inplace)
            else:
                out[comp] = value.detach().cpu().numpy()
        return out


class RandomRotation(Transform):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self):
        self.degrees = [0, 90, 180, 270]

    def __call__(self, *args):
        out = []
        degree = choice(self.degrees)
        for arg in args:
            out.append(dict())
            for comp in arg.keys():
                if isinstance(arg[comp], dict):
                    out[-1][comp] = dict()
                    for attr in arg[comp].keys():
                        if degree == 0:
                            res = arg[comp][attr]
                        elif degree == 90:
                            res = arg[comp][attr].transpose(3, 4)
                        elif degree == 180:
                            res = arg[comp][attr].flip(3)
                        elif degree == 270:
                            res = arg[comp][attr].transpose(3, 4).flip(4)
                        out[-1][comp][attr] = res
                else:
                    curr_shape = list(arg[comp].shape)
                    if len(curr_shape) >= 3:
                        if degree == 0:
                            res = arg[comp]
                        elif degree == 90:
                            res = arg[comp].transpose(-3, -2)
                        elif degree == 180:
                            res = arg[comp].flip(-3)
                        elif degree == 270:
                            res = arg[comp].transpose(-3, -2).flip(-2)
                    else:
                        res = arg[comp]
                    out[-1][comp] = res
        return tuple(out)


class Normalize(Transform):
    """Normalize samples."""
    def __init__(self, mean, std, unravel_model):
        self.mean = mean
        self.std = std
        self.unravel_model = unravel_model
        self.to_tensor = ToTensor()

    @change_sample_state(normalized=True)
    def __call__(self, sample, inplace=True):
        self._statistics_to_sample_format(sample)

        out = sample if inplace else sample.empty_like()
        spatial = out.state.spatial
        for comp, value in sample.items():
            if comp.upper() in NON_NORMALIZED_ATTRS:
                out[comp] = sample[comp]
                continue
            mask = sample.masks.actnum
            if not spatial:
                if sample.state.cropped_at_mask == 'ACTNUM':
                    mask = mask[mask == 1]
                elif sample.state.cropped_at_mask == 'WELL_MASK':
                    mask = sample.masks.well_mask[sample.masks.well_mask == 1]
                else:
                    raise ValueError('Unknown mask "%s" was used to crop the sample!' % sample.state.cropped_at_mask)
            if comp.upper() == 'CONTROL' and not sample.state.cropped_at_mask == 'WELL_MASK':
                mask = mask * sample.masks.well_mask
            if isinstance(value, BaseComponent):
                for attr, arr in value.items():
                    if attr not in self.mean[comp]:
                        continue
                    dim_diff = arr.ndim - mask.ndim
                    stats_shape = [1] * mask.ndim + [-1] + [1] * (dim_diff - 1)
                    mean = self.mean[comp][attr].reshape(stats_shape)
                    std = self.std[comp][attr].reshape(stats_shape)
                    if attr.upper() == 'DISTANCES':
                        mask = sample.masks.invalid_neighbours_mask != 1
                        out[comp][attr] = self._unitary_mean(arr, mean)
                    else:
                        std[std < 1e-3] = 1
                        out[comp][attr] = self._zero_mean_unitary_std(arr, mean, std)
                    out[comp][attr][mask == 0] = 0
            else:
                if comp not in self.mean:
                    continue
                dim_diff = value.ndim - mask.ndim
                stats_shape = [1] * (dim_diff - 1) + [-1] + [1] * mask.ndim
                mean = self.mean[comp].reshape(stats_shape)
                std = self.std[comp].reshape(stats_shape)
                std[std < 1e-3] = 1
                out[comp] = self._zero_mean_unitary_std(value, mean, std)
                out[comp][..., mask == 0] = 0
        return out

    def _statistics_to_sample_format(self, sample):
        if not self._check_is_tensor(self.mean) and self._check_is_tensor(sample):
            self.mean, self.std = self.to_tensor(self.mean), self.to_tensor(self.std)
        device = sample.device
        for comp in self.mean.keys():
            if isinstance(self.mean[comp], dict):
                for attr in self.mean[comp].keys():
                    self.mean[comp][attr] = self.mean[comp][attr].to(device)
                    self.std[comp][attr] = self.std[comp][attr].to(device)
            else:
                self.mean[comp] = self.mean[comp].to(device)
                self.std[comp] = self.std[comp].to(device)

    @staticmethod
    def _zero_mean_unitary_std(val, mean, std):
        return (val - mean) / std

    @staticmethod
    def _unitary_mean(val, mean):
        return val / mean

    def _check_is_tensor(self, x):
        """Check if x contains tensors or not."""
        for value in x.values():
            if isinstance(value, (dict, BaseComponent)):
                return self._check_is_tensor(value)
            return torch.is_tensor(value)


class Denormalize(Normalize):
    """Denormalize samples."""
    @change_sample_state(normalized=False)
    def __call__(self, sample, inplace=True):
        return super().__call__(sample, inplace=inplace)

    @staticmethod
    def _zero_mean_unitary_std(val, mean, std):
        return val * std + mean

    @staticmethod
    def _unitary_mean(val, mean):
        return val * mean


class Compose:
    """Composes several transforms together."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample, inplace=True):
        for t in self:
            sample = t(sample, inplace=inplace)
        return sample

    def __iter__(self):
        return iter(self.transforms)


class Reshape(Transform):
    """Reshape samples (crop or pad)"""
    def __init__(self, shape, pad_value=0.):
        """
        Parameters
        ----------
        shape : array_like
            New shape.
        """
        self.shape = np.array(shape)
        self.pad_value = pad_value

    def __call__(self, sample, inplace=True):
        out = sample if inplace else sample.empty_like()
        for comp, value in sample.items():
            if isinstance(value, BaseComponent):
                out[comp] = self(value)
            else:
                out[comp] = self._pad_and_crop(value, last_dims=not comp.upper() == 'XYZ')
        return out

    def _pad_and_crop(self, x, last_dims=True):
        padding, crop = self._compute_pad_and_crop(x, last_dims)
        if padding is None and crop is None:
            return x
        x = _pad_tensor(x, padding, pad_value=self.pad_value)
        x = x[crop]
        return x

    def _compute_pad_and_crop(self, x, last_dims=True):
        if len(x.shape) < len(self.shape):
            return None, None
        x_shape = np.array(x.shape)
        x_shape = x_shape[-len(self.shape):] if last_dims else x_shape[:len(self.shape)]
        diff = self.shape - x_shape

        crop_left = [-d // 2 if d < 0 else 0 for d in diff]
        crop = tuple(slice(val, val + self.shape[i]) for i, val in enumerate(crop_left))
        crop = (..., ) + crop if last_dims else crop

        padding = [(d // 2, d - d // 2) if d > 0 else (0, 0) for d in diff]
        no_padding = [(0, 0)] * (len(x.shape) - len(self.shape))
        padding = no_padding + padding if last_dims else padding + no_padding
        return padding, crop


class AddBatchDimension(Transform):
    """
    Adds a dimension corresponding to batches.
    """
    @change_sample_state(batch_dimension=True)
    def __call__(self, sample, inplace=True):
        out = sample if inplace else sample.empty_like()
        for comp, value in sample.items():
            if isinstance(value, BaseComponent):
                for attr, arr in value.items():
                    if attr.upper() == 'NAMED_WELL_MASK':
                        for well, mask in arr.items():
                            out[comp][attr][well] = mask[None]
                    elif comp.upper() in ('TABLES', 'GRID'):
                        out[comp][attr] = arr
                    else:
                        out[comp][attr] = arr[None]
            else:
                out[comp] = value[None]
        return out


class RemoveBatchDimension(Transform):
    """
    Removes a dimension corresponding to batches.
    """
    @change_sample_state(batch_dimension=False)
    def __call__(self, sample, inplace=True):
        out = sample if inplace else sample.empty_like()
        for comp, value in sample.items():
            if isinstance(value, BaseComponent):
                for attr, arr in value.items():
                    if attr.upper() == 'NAMED_WELL_MASK':
                        for well, mask in arr.items():
                            self._check_not_empty_batch(mask)
                            out[comp][attr][well] = mask[0]
                    elif comp.upper() in ('TABLES', 'GRID'):
                        out[comp][attr] = arr
                    else:
                        self._check_not_empty_batch(arr)
                        out[comp][attr] = arr[0]
            else:
                self._check_not_empty_batch(value)
                out[comp] = value[0]
        return out

    @staticmethod
    def _check_not_empty_batch(arr, dim=0):
        if arr.shape[dim] > 1:
            raise ValueError(
                """The batch can be removed only if there is 1 object in it. Found %d objects at dim %d."""
                % (arr.shape[dim], dim)
            )


class AutoPadding(Transform):
    """
    Automatically pad tensor so that dimensions are divisible by 'multipliers'.
    """
    def __init__(self, multipliers=(4, 4, 4), pad_value=0.):
        self.multipliers = np.array(multipliers)
        self.pad_value = pad_value

    def __call__(self, sample, inplace=True):
        curr_shape = self._get_input_shape(sample)
        if curr_shape is None:
            return sample

        new_shape = self._get_new_shape(curr_shape)
        if np.all(new_shape == curr_shape):
            return sample

        return Reshape(shape=new_shape, pad_value=self.pad_value)(sample, inplace)

    def _get_new_shape(self, curr_shape):
        return np.array([np.ceil(d/m)*m for d, m in zip(curr_shape, self.multipliers)]).astype(int)

    def _get_input_shape(self, sample):
        for comp, value in sample.items():  # pylint: disable=too-many-nested-blocks
            if isinstance(value, BaseComponent):
                for arr in value.values():
                    if isinstance(arr, BaseComponent):
                        for mask in arr.values():
                            if len(mask.shape) >= len(self.multipliers):
                                return mask.shape[-len(self.multipliers):]
                    if len(arr.shape) >= len(self.multipliers):
                        return arr.shape[-len(self.multipliers):]
            else:
                if len(value.shape) >= len(self.multipliers):
                    if comp.upper() != 'GRID':
                        return value.shape[-len(self.multipliers):]
                    return value.shape[:len(self.multipliers)]
        return None


def _pad_tensor(tensor, padding, pad_value=0.):
    if isinstance(tensor, torch.Tensor):
        tensor = F.pad(tensor,
                       [item for sublist in reversed(padding) for item in sublist],
                       mode='constant',
                       value=pad_value
                       )
    elif isinstance(tensor, np.ndarray):
        padding = [(0, 0)]*(tensor.ndim - len(padding)) + list(padding)
        tensor = np.pad(tensor, padding, mode='constant')
    else:
        raise ValueError('`tensor` should be of type numpy.ndarray or torch.Tensor, not {}'.format(type(tensor)))
    return tensor
