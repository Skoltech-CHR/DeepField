"""Base class for learnable models."""
import torch
from torch import nn

from .utils import get_model_device
from ..field.utils import hasnested


class BaseModel(nn.Module):
    """Base-class for neural-network models.
    Contains required API for model training, dump and load procedures."""
    def forward(self, *args, **kwargs):
        """Forward pass."""
        raise NotImplementedError('Abstract method is not implemented.')

    def make_training_iter(self, sample, loss_pattern, **kwargs):
        """Make one training iteration. Runs backward pass. Return loss of the prediction.
        Loss should be detached."""
        raise NotImplementedError('Abstract method is not implemented.')

    def make_evaluation_iter(self, sample, loss_pattern, **kwargs):
        """Make evaluation iteration.
        Wraps self.make_training_iter with torch.no_grad wrapper."""
        is_training = self.training
        self.eval()
        with torch.no_grad:
            loss = self.make_training_iter(sample, loss_pattern, **kwargs)
        self.train(mode=is_training)
        return loss

    def load(self, path):
        """Load model parameters from the file.

        Parameters
        ----------
        path: str
            Path to the file with stored parameters.

        """
        loaded_dict = torch.load(path, map_location=get_model_device(self))
        self.load_state_dict(loaded_dict)

    def dump(self, path):
        """Dump model parameters into the file.

        Parameters
        ----------
        path: str
            Path to the file for dump.

        """
        torch.save(self.state_dict(), path)

    def _get_attrs_from_sample(self, sample, *attrs, leave_attrs_device=()):
        """Get attributes from sample by name.
        Move attributes to model's device unless they are in `leave_attrs_device`.

        Parameters
        ----------
        sample
        attrs
        leave_attrs_device

        Returns
        -------
        out: tuple
        """
        device = get_model_device(self)
        out = ()

        for attr in attrs:
            # FIXME attr is not str - it is tuple! Values are always on the model's device!
            value = sample[attr] if isinstance(attr, str) else sample[attr[0]][attr[1]]
            out += (value.to(device) if attr not in leave_attrs_device else value, )

        if hasnested(sample, 'masks', 'neighbours'):
            value = sample['masks']['neighbours'].long()
            if ('masks', 'neighbours') not in leave_attrs_device:
                value = value.to(device)
            out += (value, )
            if hasnested(sample, 'grid', 'distances'):
                value = sample['grid']['distances'].long()
                if ('grid', 'distances') not in leave_attrs_device:
                    value = value.to(device)
                out += (value, )
        return out

    def _get_masks_from_sample(self, sample, *mask_keys):
        """Get masks from sample by name.

        Parameters
        ----------
        sample
        mask_keys

        Returns
        -------
        out: tuple
        """
        device = get_model_device(self)
        masks = ()
        for mask in mask_keys:
            if mask is None:
                masks += (mask, )
            elif hasnested(sample, 'masks', mask):
                masks += (sample['masks'][mask].to(device), )
            else:
                raise ValueError('Mask key %s was not found in the sample' % mask)
        return masks

    @staticmethod
    def _compute_loss(ref, pred, loss_pattern, masks):
        """Compute loss between reference and prediction, given `loss_pattern`.
        If `loss_pattern` contains masked losses, the masks should be also provided.

        Parameters
        ----------
        ref
        pred
        loss_pattern
        masks

        Returns
        -------
        loss
        """
        loss = [0]
        for mask, pattern in zip(masks, loss_pattern):
            mult, loss_fn = pattern['multiplier'], pattern['loss_fn']
            kwargs = pattern['kwargs'] if 'kwargs' in pattern else {}
            loss.append(mult * loss_fn(ref=ref, pred=pred, mask=mask, **kwargs))
        return sum(loss)
