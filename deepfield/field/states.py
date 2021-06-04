"""States component."""
import os
import glob
import numpy as np

from .decorators import apply_to_each_input, state_check, ndim_check
from .base_spatial import SpatialComponent
from .plot_utils import show_cube_static, show_cube_interactive
from .parse_utils import read_ecl_bin
from .utils import get_single_path

FULL_STATE_KEYS = ('PRESSURE', 'RS', 'SGAS', 'SOIL', 'SWAT')


class States(SpatialComponent):
    """States component of geological model."""

    @property
    def n_timesteps(self):
        """Effective number of timesteps."""
        if not self.attributes:
            return 0
        return np.min([x.shape[0] for _, x in self.items()])

    @apply_to_each_input
    def apply(self, func, attr, *args, inplace=False, **kwargs):
        """Apply function to each timestamp of states attributes.

        Parameters
        ----------
        func : callable
            A function to apply. Must accept data as its first argument.
        attr : str, array-like
            Attributes to get data from.
        args : misc
            Any additional positional arguments to ``func``.
        inplace: bool
            Modify сomponent inplace.
        kwargs : misc
            Any additional named arguments to ``func``.

        Returns
        -------
        output : States
            Transformed component.
        """
        data = getattr(self, attr)
        res = np.array([func(x, *args, **kwargs) for x in data])
        if inplace:
            setattr(self, attr, res)
            return self
        return res

    @apply_to_each_input
    def _to_spatial(self, attr, dimens, inplace):
        """Spatial order 'F' transformations."""
        return self.reshape(attr=attr, newshape=(self.n_timesteps,) + tuple(dimens),
                            order='F', inplace=inplace)

    @apply_to_each_input
    def _ravel(self, attr, inplace):
        """Ravel order 'F' transformations."""
        return self.reshape(attr=attr, newshape=(self.n_timesteps, -1), order='F', inplace=inplace)

    @apply_to_each_input
    def pad_na(self, attr, actnum, fill_na=0., inplace=True):
        """Add dummy cells into the state vector in the positions of non-active cells if necessary.

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
        if np.prod(data.shape[1:]) == actnum.size:
            return self if inplace else data
        if data.ndim > 2:
            raise ValueError('Data should be raveled before padding.')
        n_ts = data.shape[0]

        actnum_ravel = actnum.ravel(order='F').astype(bool)
        not_actnum_ravel = ~actnum_ravel
        padded_data = np.empty(shape=(n_ts, actnum.size), dtype=float)
        padded_data[..., actnum_ravel] = data
        del data
        padded_data[..., not_actnum_ravel] = fill_na

        if inplace:
            setattr(self, attr, padded_data)
            return self
        return padded_data

    @apply_to_each_input
    def strip_na(self, attr, actnum, inplace=True):
        """Remove non-active cells from the state vector.

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

        Notes
        -----
        Outputs 1d array for each timestamp.
        """
        if self.state.spatial and inplace:
            raise ValueError('Inplace is not allowed in spatial state.')
        data = self.ravel(attr, inplace=False)
        if data.shape[1] == np.sum(actnum):
            return self if inplace else data
        stripped_data = data[..., actnum.ravel(order='F')]
        if inplace:
            setattr(self, attr, stripped_data)
            return self
        return stripped_data

    def __getitem__(self, keys):
        out = self.__class__()
        for attr, data in self.items():
            data = data[keys].reshape((-1,) + data.shape[1:])
            setattr(out, attr, data)
        out.set_state(**self.state.as_dict())
        return out

    @state_check(lambda state: state.spatial)
    @ndim_check(4)
    def show_cube(self, attr, t=None, x=None, y=None, z=None, actnum=None, figsize=None, **kwargs):
        """Visualize slices of 4D states arrays. If no slice is specified, spatial slices
        will be shown with interactive slider widgets.

        Parameters
        ----------
        attr : str
            Attribute to show.
        t : int or None, optional
            Timestamp to show.
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
        if np.all([t is None, x is None, y is None, z is None]):
            show_cube_interactive(data, figsize=figsize, **kwargs)
        else:
            show_cube_static(data, t=t, x=x, y=y, z=z, figsize=figsize, **kwargs)
        return self

    def _read_buffer(self, path_or_buffer, attr, **kwargs):
        super()._read_buffer(path_or_buffer, attr, **kwargs)
        return self.reshape(attr=attr, newshape=(1, -1))

    def _load_ecl_binary(self, path_to_results, attrs, basename, logger=None, **kwargs):
        """Load states from binary ECLIPSE results files.

        Parameters
        ----------
        path_to_results : str
            Path to the folder with precomputed results of hydrodynamical simulation
        attrs : list or str
            Keyword names to be loaded
        logger : logger
            Logger for messages.
        **kwargs : dict, optional
            Any kwargs to be passed to load method.

        Returns
        -------
        states : States
            States with loaded attributes.
        """
        if attrs is None:
            attrs = FULL_STATE_KEYS
        unifout_path = get_single_path(path_to_results, basename + '.UNRST', logger)
        multout_paths = _get_multout_paths(path_to_results)
        if unifout_path is not None:
            return self._load_ecl_bin_unifout(unifout_path, attrs=attrs, logger=logger, **kwargs)
        if multout_paths is not None:
            return self._load_ecl_bin_multout(multout_paths, attrs=attrs, logger=logger, **kwargs)
        if logger is not None:
            logger.warning('The results in "%s" were not found!' % path_to_results)
            return self
        raise FileNotFoundError('The results in "%s" were not found!' % path_to_results)

    def _load_ecl_bin_unifout(self, path, attrs, logger, subset=None, **kwargs):
        """Load states from .UNRST binary file.

        Parameters
        ----------
        path: str
            Path to the .UNRST file.
        attrs: list or str
            Keyword names to be loaded from the file.
        kwargs : dict, optional
            Any kwargs to be passed to load method.

        Returns
        -------
        states : States
            States with loaded attributes.
        """
        _ = kwargs
        if isinstance(attrs, str):
            attrs = [attrs]
        states = read_ecl_bin(path, attrs, sequential=True, subset=subset, logger=logger)
        for attr, x in states.items():
            setattr(self, attr, np.array(x))
        return self

    def _load_ecl_bin_multout(self, paths, attrs, logger, subset=None, **kwargs):
        """Load states from .X____ binary files.

        Parameters
        ----------
        paths: list
            List of paths to .X____ files
        attrs: list or str
            Keyword names to be loaded from the files.
        kwargs : dict, optional
            Any kwargs to be passed to load method.

        Returns
        -------
        states : States
            States with loaded attributes.
        """
        _ = kwargs
        if isinstance(attrs, str):
            attrs = [attrs]
        states = {}
        logger.info('Start reading X files.')

        def is_in_subset(x):
            fmt = os.path.splitext(x)[1]
            timestep = int(fmt.lstrip('.X'))
            criteria = timestep in subset
            return criteria

        paths = filter(is_in_subset if subset is not None else None, paths)
        for path in paths:
            state = read_ecl_bin(path, attrs, logger=logger)
            for attr, x in state.items():
                if attr not in states:
                    states[attr] = []
                else:
                    states[attr].append(x)
        logger.info('Finish reading X files.')
        states = {attr: np.stack(x) for attr, x in states.items()}
        for attr, x in states.items():
            setattr(self, attr, x)
        return self

    def _make_data_dump(self, attr, fmt=None, actnum=None, float_dtype=None, **kwargs):
        """Prepare data for dump."""
        if fmt.upper() == 'ASCII':
            data = self.ravel(attr=attr, inplace=False)
            return data[0]
        if fmt.upper() == 'HDF5':
            if actnum is None:
                data = self.ravel(attr=attr, inplace=False)
            else:
                data = self.strip_na(attr=attr, actnum=actnum, inplace=False)
            return data if float_dtype is None else data.astype(float_dtype)
        return super()._make_data_dump(attr, fmt=fmt, **kwargs)


def _get_multout_paths(path_to_results):
    """Searches for .X____ files in a RESULT folder near .DATA file

    Parameters
    ----------
    path_to_results: str
        Path to the folder with precomputed results of hydrodynamical simulation

    Returns
    -------
    paths: list or None
        List of the paths found. None else.
    """
    multout_paths = glob.glob(os.path.join(path_to_results, '*.X*'))
    return sorted(multout_paths) if multout_paths else None
