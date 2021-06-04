"""Rock component."""
import warnings
import numpy as np
import matplotlib.pyplot as plt

from .base_spatial import SpatialComponent
from .decorators import apply_to_each_input, state_check, ndim_check
from .plot_utils import show_cube_static, show_cube_interactive
from .utils import rolling_window, get_single_path
from .parse_utils import read_ecl_bin


class Rock(SpatialComponent):
    """Rock component of geological model."""

    def _load_ecl_binary(self, path_to_results, attrs, basename, logger=None):
        path = get_single_path(path_to_results, basename + '.INIT', logger)
        if path is None:
            return
        sections = read_ecl_bin(path, attrs, logger=logger)
        for k in ['PORO', 'PERMX', 'PERMY', 'PERMZ']:
            if (k in attrs) and (k in sections):
                setattr(self, k, sections[k])

    @apply_to_each_input
    def _to_spatial(self, attr, dimens, inplace):
        """Spatial order 'F' transformations."""
        return self.reshape(attr=attr, newshape=dimens, order='F', inplace=inplace)

    def _make_data_dump(self, attr, fmt=None, float_dtype=None, **kwargs):
        """Prepare data for dump."""
        if fmt.upper() != 'HDF5':
            return super()._make_data_dump(attr, fmt=fmt, **kwargs)
        data = self.ravel(attr=attr, inplace=False)
        return data if float_dtype is None else data.astype(float_dtype)

    @apply_to_each_input
    def pad_na(self, attr, actnum, fill_na=0., inplace=True):
        """Add dummy cells into the rock vector in the positions of non-active cells if necessary.

        Parameters
        ----------
        attr: str, array-like
            Attributes to be padded with non-active cells.
        actnum: array-like of type bool
            Vector representing a mask of active and non-active cells.
        fill_na: float
            Value to be used as filler.
        inplace: bool
            Modify сomponent inplace.

        Returns
        -------
        output : component if inplace else padded attribute.
        """
        data = getattr(self, attr)
        if np.prod(data.shape) == actnum.size:
            return self if inplace else data
        if data.ndim > 1:
            raise ValueError('Data should be raveled before padding.')
        padded_data = np.full(shape=(actnum.size,), fill_value=fill_na, dtype=float)
        padded_data[actnum.ravel(order='F')] = data
        if inplace:
            setattr(self, attr, padded_data)
            return self
        return padded_data

    @apply_to_each_input
    def strip_na(self, attr, actnum, inplace=True):
        """Remove non-active cells from the rock vector.

        Parameters
        ----------
        attr: str, array-like
            Attributes to be stripped
        actnum: array-like of type bool
            Vector representing mask of active and non-active cells.
        inplace: bool
            Modify сomponent inplace.

        Returns
        -------
        output : component if inplace else stripped attribute.
        """
        if self.state.spatial and inplace:
            raise ValueError('Inplace is not allowed in spatial state.')
        data = self.ravel(attr, inplace=False)
        if data.size == np.sum(actnum):
            return self if inplace else data
        stripped_data = data[actnum.ravel(order='F')]
        if inplace:
            setattr(self, attr, stripped_data)
            return self
        return stripped_data

    def show_histogram(self, attr, actnum=None, **kwargs):
        """Show properties distribution.

        Parameters
        ----------
        attr : str
            Attribute to compute the histogram.
        actnum : array, optional
            Actnum array. If None, all cell are active.
        kwargs : misc
            Any additional named arguments to ``plt.hist``.

        Returns
        -------
        plot : Histogram plot.
        """
        data = getattr(self, attr)
        if actnum is not None:
            data = data * actnum
        plt.hist(data.ravel(), **kwargs)
        plt.show()
        return self

    @state_check(lambda state: state.spatial)
    @ndim_check(3)
    def show_cube(self, attr, x=None, y=None, z=None, actnum=None, figsize=None, **kwargs):
        """Visualize slices of 3D array. If no slice is specified, all 3 slices
        will be shown with interactive slider widgets.

        Parameters
        ----------
        attr : str
            Attribute to show.
        x : int or None, optional
            Slice along x-axis to show.
        y : int or None, optional
            Slice along y-axis to show.
        z : int or None, optional
            Slice along z-axis to show.
        actnum : array, optional
            Actnum array. If None, all cell are active.
        figsize : array-like, optional
            Output plot size.
        kwargs : dict, optional
            Additional keyword arguments for plot.
        """
        data = getattr(self, attr)
        if actnum is not None:
            data = data * actnum
        if np.all([x is None, y is None, z is None]):
            show_cube_interactive(data, figsize=figsize, **kwargs)
        else:
            show_cube_static(data, x=x, y=y, z=z, figsize=figsize, **kwargs)
        return self

    @state_check(lambda state: state.spatial)
    @apply_to_each_input
    def upscale(self, attr, factors, volumes, weights=None):
        """Upscale properties.

        Parameters
        ----------
        attr : str, list of str
            Attributes to be upscaled.
        factors : tuple, int
            Scale factors along each axis. If int, factors are the same for each axis.
        volumes : ndarray, same shape as `attr`
            Cell volumes.
        weights : ndarray, same shape as `attr`, optional
            Cell weights.

        Returns
        -------
        out : ndarray
            Upscaled properties.
        """
        factors = np.atleast_1d(factors)
        if factors.size == 1:
            factors = np.repeat(factors, 3)
        data = getattr(self, attr)
        binned_weights = rolling_window(volumes, factors)
        if weights is not None:
            binned_weights *= rolling_window(weights, factors)
        out = rolling_window(data, factors) * binned_weights
        binned_weights = np.sum(binned_weights, axis=(-3, -2, -1))
        zero_mask = np.isclose(binned_weights, 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = np.sum(out, axis=(-3, -2, -1)) / binned_weights
        out[zero_mask] = 0
        return out
