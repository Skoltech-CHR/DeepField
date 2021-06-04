"""SpatialComponent class."""
import numpy as np
import skimage
from skimage.transform import rescale, resize
import scipy
from deprecated.sphinx import deprecated

from .base_component import BaseComponent
from .decorators import apply_to_each_input, add_actions, extract_actions, TEMPLATE_DOCSTRING

ACTIONS_DICT = {
    "pad": (np.pad, "numpy.pad", "padded array"),
    "flip": (np.flip, "numpy.flip", "reversed order of elements in an array along the given axis"),
    "clip": (np.clip, "numpy.clip", "array of cliped values"),
    "rot90": (np.rot90, "numpy.rot90", "rotated an array by 90 degrees in the plane specified by axes"),
    "gradient": (np.gradient, "numpy.gradient", "gradient"),
    "resize": (resize, "skimage.transform.resize", "resize"),
    "rescale": (rescale, "skimage.transform.rescale", "rescale"),
    "crop": (skimage.util.crop, "crop", "cropped array by crop_width along each dimension"),
    "random_noise": (skimage.util.random_noise, "random_noise",
                     "array with added random noise of various types"),
}

@add_actions(extract_actions(scipy.ndimage), TEMPLATE_DOCSTRING)
@add_actions(ACTIONS_DICT, TEMPLATE_DOCSTRING)
class SpatialComponent(BaseComponent):
    """Base component for spatial-type attributes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_state(spatial=None)

    def sample_crops(self, attr, shape, size=1):
        """Sample random crops of fixed shape.

        Parameters
        ----------
        attr : str, array-like
            Attributes to sample crops from. If None, use all attributes.
        shape : tuple
            Shape of crops.
        size : int, optional
            Number of crops to sample. Default to 1.

        Returns
        -------
        crops : ndarray
            Sampled crops.
        """
        is_list = True
        if isinstance(attr, str):
            attr = [attr]
            is_list = False
        if attr is None:
            attr = self.attributes
        data_shape = np.array(getattr(self, attr[0]).shape)
        valid_range = data_shape - np.array(shape)
        before = np.array([np.random.randint(0, d, size=size) for d in valid_range]).T
        after = valid_range - before
        res = [self.crop(attr=attr, crop_width=list(zip(before[i], after[i]))) for i in range(size)]
        res = np.swapaxes(res, 0, 1)
        return res[0] if is_list else res

    def ravel(self, attr=None, inplace=True, **kwargs):
        """Brings component to ravel state. If not inplace returns
        ravel representation for attributes with pre-defined ravel transformation.

        Parameters
        ----------
        attr : str, array of str
            Attribute to ravel.
        inplace : bool
            Modify сomponent inplace.
        kwargs : misc
            Additional named arguments.

        Returns
        -------
        out : component if inplace else raveled attribute.
        """
        if attr is not None and inplace:
            raise ValueError('`attr` should be None for inplace operation.')
        res = self._ravel(attr=attr, inplace=inplace, **kwargs)
        if not inplace:
            return res
        self.set_state(spatial=False)
        return self

    def _ravel(self, attr, inplace, **kwargs):
        """Ravel transformations."""
        return super().ravel(attr=attr, inplace=inplace, **kwargs)

    def to_spatial(self, attr=None, inplace=True, **kwargs):
        """Bring component to spatial state. If not inplace returns
        spatial representation for attributes with pre-defined spatial transformation.

        Parameters
        ----------
        attr : str, array of str
            Attribute to ravel.
        inplace : bool
            Modify сomponent inplace.
        kwargs : misc
            Additional named arguments.

        Returns
        -------
        out : сomponent if inplace else raveled attribute.
        """
        if attr is not None and inplace:
            raise ValueError('`attr` should be None for inplace operation.')
        res = self._to_spatial(attr=attr, inplace=inplace, **kwargs)
        if not inplace:
            return res
        self.set_state(spatial=True)
        return self

    @apply_to_each_input
    def _to_spatial(self, attr, inplace, **kwargs):
        """Spatial transformations."""
        _ = attr, inplace, kwargs
        raise NotImplementedError()

    @deprecated(reason="Renamed to_spatial")
    def unravel(self, *args, **kwargs):
        """Alias for `to_spatial` method."""
        return self.to_spatial(*args, **kwargs)

    def _make_data_dump(self, attr, fmt=None, **kwargs):
        _ = fmt, kwargs
        return self.ravel(attr=attr, order='F', inplace=False)
