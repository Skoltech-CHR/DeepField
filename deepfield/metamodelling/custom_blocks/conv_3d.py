"""Custom blocks for 3D convolutional models."""
from torch import nn
from torch.nn.modules.utils import _triple


class CenterCrop3d(nn.Module):
    """CenterCrop for 3d tensors."""
    def __init__(self, crop):
        """
        Parameters
        ----------
        crop: int, tuple
            The amount of cells to crop from each axis.
        """
        super().__init__()
        self.crop = _triple(crop)

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Tensor to crop.
            Shape: [B, C, H, W, D]

        Returns
        -------
        out: torch.Tensor
            Cropped tensor.
            Shape: [B, C, H - 2 * crop[0], W - 2 * crop[1], D - 2 * crop[2]]
        """
        i, j, k = self.crop
        return x[:, :, i:-i, j:-j, k:-k]


class ZeroPad3d(nn.ConstantPad3d):
    """Padding with zeros for 3d tensors."""
    def __init__(self, padding):
        """
        Parameters
        ----------
        padding: int, tuple
            The amount of cells to add to each axis.
        """
        super().__init__(padding, 0.)


class VoxelShuffle(nn.Module):
    """Modification of PixelShuffle (arXiv:1609.05158) for 3d tensors.
    Expand spatial dimensions `upscale_factor` times at the expense of number of channels.
    Number of channels should be divisible by the `upscale_factor` ** 3 before the operation."""
    def __init__(self, upscale_factor):
        """
        Parameters
        ----------
        upscale_factor
        """
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        """
        Parameters
        ----------
        x: torch.Tensor
            Tensor to be transformed with VoxelShuffle.
            Shape: [B, C, H, W, D]
            `C` should be divisible by the `upscale_factor`.


        Returns
        -------
        out: torch.Tensor
            Transformed tensor.
            Shape: [B, C / (uf ** 3), H * uf, W * uf, D * uf]
            where `uf` is the `upscale_factor`.
        """
        return self._voxel_shuffle(x, self.upscale_factor)

    @staticmethod
    def _voxel_shuffle(x, upscale_factor):
        b, c, h, w, d = x.size()

        c //= upscale_factor ** 3
        x_reshaped = x.reshape(b, c, upscale_factor, upscale_factor, upscale_factor, h, w, d)

        h *= upscale_factor
        w *= upscale_factor
        d *= upscale_factor
        return x_reshaped.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(b, c, h, w, d)
