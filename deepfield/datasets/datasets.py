# pylint: disable=too-many-lines
"""Dataset wrappers for Fields."""
import os
import pickle
import inspect
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from ..field import Field
from ..field.base_component import BaseComponent
from ..field.utils import recursive_insensitive_glob, hasnested, overflow_safe_mean, get_spatial_perf
from .utils import get_config, STATES_KEYWORD
from .transforms import ToNumpy, Normalize, Compose, RemoveBatchDimension, AddBatchDimension, \
    Transform, NON_NORMALIZED_ATTRS

SEQUENTIAL_ATTRS = ['STATES', 'CONTROL']
TABLES_WITHOUT_INDEX = ['DENSITY']
INVALID_VALUE_FILLER = -1
CONTROL_TO_RESULTS_KW = {'WBHP': 'BHPT'}


def safe_check(comp, state, expected, default=False):
    """Check that components's state has expected value or return default if state is not defined."""
    try:
        return getattr(comp.state, state) == expected
    except AttributeError:
        return default


class FieldDataset(Dataset):  # pylint: disable=too-many-instance-attributes
    """Baseclass for dataset of fields loaded with similar configs."""
    default_sample_attrs = {
                'MASKS': ['ACTNUM', 'TIME'],
                'GRID': [],
                'ROCK': ['PORO', 'PERMX', 'PERMY', 'PERMZ'],
                'STATES': ['PRESSURE', 'RS', 'SGAS', 'SOIL', 'SWAT'],
                'CONTROL': ['BHPT'],
    }

    _attrs_sampled_as_dict = ('MASKS', 'GRID', 'TABLES')

    def __init__(self, src, sample_attrs=None, fmt=('dat', 'data', 'hdf5'), subset_generator=None,
                 unravel_model=None, from_samples=False, allow_change_preloaded=False):
        """
        Parameters
        ----------
        src: str, Field, FieldSample or list of Fields or FieldSamples
            Path to a directory containing fields for the dataset or preloaded Fields
        sample_attrs: dict
            Attributes to be represented in samples
        fmt: str or tuple
            Format in which fields are represented
        subset_generator: callable or None
            Function generating subsequences for sequential attrs (states, control)
            Should return array-like objects with timestep indices
            If None, full sequences will be sampled
        unravel_model: bool or None
            Either or not unravel loaded models
            If None, will be inferred from sample_attrs (set to False if 'neighbours' or 'distances' keys are presented)
        from_samples: bool
            If True, tries to load samples from previously dumped dataset (with FieldDataset.dump_samples).
            The sample_attrs will not affect the content of the loaded samples.
            The transforms will still be applied.
        """
        # TODO: add possibility to make subsets of timesteps limited to constant control
        super().__init__()
        if isinstance(fmt, str):
            fmt = (fmt, )
        files = []
        self.root_dir = None
        self.preloaded = None
        if isinstance(src, str):
            for f in fmt:
                files += recursive_insensitive_glob(src, pattern='*.%s' % f, return_relative=True)
            self.root_dir = src
        else:
            self.preloaded = np.atleast_1d(src)
        self.fmt = fmt
        self.files = files
        self.transform = None
        self._sample_attrs = None
        self.sample_attrs = sample_attrs if sample_attrs is not None else self.default_sample_attrs
        self.from_samples = from_samples
        self.allow_change_preloaded = allow_change_preloaded

        self.config = get_config()
        # TODO make config dependent on the sample attrs
        self.subset_generator = subset_generator

        self.mean = None
        self.std = None
        self.min = None
        self.max = None

        self.masks_getter_map = {
            'ACTNUM': self._get_actnum,
            'WELL_MASK': self._get_well_mask,
            'NAMED_WELL_MASK': self._get_named_well_mask,
            'NEIGHBOURS': self._get_neighbours,
            'INVALID_NEIGHBOURS_MASK': self._get_invalid_neighbours_mask,
            'TIME': self._get_time,
            'CF_MASK': self._get_connection_factors,
            'PERF_MASK': self._get_perforation_mask
        }
        self.grid_getter_map = {
            'DISTANCES': self._get_distances,
            'XYZ': self._get_xyz
        }
        self.attrs_getter_map = {
            'STATES': self._get_states,
            'ROCK': self._get_rock,
            'CONTROL': self._get_control
        }

        invalid_unravel_attrs = {
            'MASKS': ['NEIGHBOURS'],
            'GRID': ['DISTANCES']
        }
        if unravel_model:
            for comp, attrs in invalid_unravel_attrs.items():
                for attr in attrs:
                    if hasnested(self.sample_attrs, comp, attr):
                        raise ValueError('Can not unravel model and sample %s simultaneously.' % attr)
        if unravel_model is None:
            ravel = False
            for comp, attrs in invalid_unravel_attrs.items():
                for attr in attrs:
                    ravel = ravel or hasnested(self.sample_attrs, comp, attr)
            unravel_model = not ravel
        self.unravel_model = unravel_model

    def __len__(self):
        if self.preloaded is not None:
            return len(self.preloaded)
        return len(self.files)

    def __getitem__(self, idx):
        if self.from_samples:
            sample = self._load_sample(idx)
        else:
            sample = self._get_sample(idx)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def _get_sample(self, idx):  # pylint: disable=too-many-branches
        """Get sample from the dataset

        Parameters
        ----------
        idx: int, torch.Tensor
            Index of the field

        Returns
        -------
        sample: FieldSample
        """
        # TODO time and batch dimensions are transposed
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.subset_generator is not None:
            sequence_subset = list(self.subset_generator())
            if not sequence_subset:
                raise ValueError('subset_generator should not generate empty subsets!')
        else:
            sequence_subset = None
        config = self.config.copy()
        config[STATES_KEYWORD] = {'attrs': self.config[STATES_KEYWORD]['attrs'], 'subset': sequence_subset}
        if 'Aquifers' in config:
            del config['Aquifers']
        if self.preloaded is None:
            model = self._load_model(idx, config)
        else:
            if isinstance(self.preloaded[idx], FieldSample):
                return self.preloaded[idx]
            model = self._get_preloaded(idx)

        sample = {}
        getter_kwargs = dict(
            sequence_subset=sequence_subset, fill_invalid_neighbours=INVALID_VALUE_FILLER, neighbouring_radius=1
        )

        for comp, attrs in self.sample_attrs.items():
            if comp == 'MASKS':
                sample[comp] = {}
                for attr, mask_getter in self.masks_getter_map.items():
                    # FIXME
                    if not self.unravel_model and attr.upper() in ('CF_MASK', 'PERF_MASK'):
                        continue
                    sample[comp][attr] = mask_getter(model, **getter_kwargs)
            elif comp == 'GRID':
                sample[comp] = {}
                for attr in attrs:
                    sample[comp][attr] = self.grid_getter_map[attr](model, **getter_kwargs)
            elif comp == 'TABLES':
                sample[comp] = {}
                sample[comp] = self._get_tables(model, attrs)
            elif comp == 'CONTROL':
                res = self.attrs_getter_map[comp](model, attrs, **getter_kwargs)
                sample[comp] = res['control']
                sample['MASKS']['CONTROL_T'] = res['t']
            else:
                sample[comp] = self.attrs_getter_map[comp](model, attrs, **getter_kwargs)

        sample = FieldSample(field=model, dataset=self, **sample)
        for key in list(sample.masks.keys()):
            if sample.masks[key] is None:
                del sample.masks[key]
        if not self.unravel_model:
            sample.as_ravel(inplace=True)
        return sample

    def _get_preloaded(self, idx):
        """Get a field from preloaded."""
        model = self.preloaded[idx]
        if self.allow_change_preloaded:
            if model.state.spatial != self.unravel_model:
                if self.unravel_model:
                    print('to_spatial')
                    model.to_spatial()
                else:
                    print('ravel')
                    model.ravel()
            if 'CONTROL' in self.sample_attrs:
                if not model.wells.state.all_tracks_complete:
                    model.wells.drop_incomplete()
                if not model.wells.state.has_blocks:
                    model.wells.get_wellblocks(model.grid)
                if not model.wells.state.full_perforation:
                    model.wells.apply_perforations()
                if not model.wells.state.all_tracks_inside:
                    model.wells.drop_outside()
                if model.meta['MODEL_TYPE'] == 'ECL':
                    model.wells.compute_events(grid=model.grid)
        else:
            assert model.state.spatial == self.unravel_model
            if 'CONTROL' in self.sample_attrs:
                assert model.wells.state.all_tracks_complete
                assert model.wells.state.has_blocks
                assert model.wells.state.full_perforation
                assert model.wells.state.all_tracks_inside
        return model

    def _load_model(self, idx, config=None, force_wells_calculations=False):
        """Loads field by index.

        Parameters
        ----------
        idx: int
        config: dict, optional
            Config used while loading the model

        Returns
        -------
        model: Field
        """
        _, fmt = os.path.splitext(self.files[idx])
        fmt = fmt.strip('.').lower()

        config = self.config if config is None else config

        if fmt == 'hdf5':
            if 'subset' not in config[STATES_KEYWORD] or config[STATES_KEYWORD]['subset'] is None:
                config = None
            else:
                for comp in config:
                    config[comp]['attrs'] = None

        model = Field(path=os.path.join(self.root_dir, self.files[idx]), config=config,
                      encoding='auto:10000', loglevel='ERROR')
        model.load(raise_errors=False)

        if self.unravel_model:
            model.to_spatial()
        if 'CONTROL' in self.sample_attrs:
            if not safe_check(model.wells, 'all_tracks_complete', True) or force_wells_calculations:
                model.wells.drop_incomplete()
            if not safe_check(model.wells, 'has_blocks', True) or force_wells_calculations:
                model.wells.get_wellblocks(grid=model.grid)
            if not safe_check(model.wells, 'full_perforation', True) or force_wells_calculations:
                model.wells.apply_perforations()
            if not safe_check(model.wells, 'all_tracks_inside', True) or force_wells_calculations:
                model.wells.drop_outside()
            if model.meta['MODEL_TYPE'] == 'ECL':
                model.wells.compute_events(grid=model.grid)
        if not self.unravel_model:
            model.ravel()
        return model

    def _load_sample(self, idx):
        sample = FieldSample(os.path.join(self.root_dir, self.files[idx]))
        sample.load()
        return sample

    def _get_actnum(self, model, **kwargs):
        """Get ACTNUM of the model"""
        _ = kwargs
        if hasattr(model.grid, 'actnum'):
            return getattr(model.grid, 'actnum').astype(np.bool)
        actnum = np.ones(model.grid.dimens, dtype=np.bool)
        return actnum if self.unravel_model else actnum.ravel(order='F')

    def _get_well_mask(self, model, **kwargs):
        """Get well mask of the model."""
        _ = kwargs
        if hasnested(self.sample_attrs, 'MASKS', 'WELL_MASK') or 'CONTROL' in self.sample_attrs:
            return model.well_mask != ''
        return None

    def _get_named_well_mask(self, model, **kwargs):
        """Get well mask of the model."""
        _ = kwargs
        if hasnested(self.sample_attrs, 'MASKS', 'NAMED_WELL_MASK') or 'CONTROL' in self.sample_attrs:
            well_mask = model.well_mask
            named_well_mask = {}
            for well in model.wells:
                named_well_mask[well.name] = well_mask == well.name
            return named_well_mask
        return None

    def _get_neighbours(self, model, fill_invalid_neighbours=INVALID_VALUE_FILLER, neighbouring_radius=-1, **kwargs):
        """Get connectivity matrix of cells presented in the model."""
        if 'MASKS' in self.sample_attrs and 'NEIGHBOURS' in self.sample_attrs['MASKS']:
            neighbours = model.grid.get_neighbors_matrix(
                connectivity=neighbouring_radius,
                fill_value=fill_invalid_neighbours,
                ravel_index=True
            )
            # Indices are with respect to the full vectors: with active and non-active cells
            # We want indices with respect to the vector of active cells
            full_to_active_ind = kwargs['MASKS']['ACTNUM'].copy().astype(np.int)
            full_to_active_ind[full_to_active_ind == 1] = np.arange(full_to_active_ind.sum())
            full_to_active_ind = np.concatenate([full_to_active_ind, [-1]])
            neighbours = full_to_active_ind[neighbours.ravel()].reshape(neighbours.shape)
            # Neighbours should include the cell itself
            itself_ind = np.arange(neighbours.shape[0])[:, np.newaxis]
            neighbours = np.concatenate([itself_ind, neighbours], axis=1)
            return neighbours
        return None

    def _get_invalid_neighbours_mask(self, model, fill_invalid_neighbours=INVALID_VALUE_FILLER,
                                     neighbouring_radius=-1, **kwargs):
        """Get mask of invalid neighbours (non-active or out of geometric bounds)."""
        _ = kwargs
        if hasnested(self.sample_attrs, 'GRID', 'DISTANCES'):
            neighbours = model.grid.get_neighbors_matrix(
                connectivity=neighbouring_radius,
                fill_value=fill_invalid_neighbours,
                ravel_index=True
            )
            return neighbours == INVALID_VALUE_FILLER
        return None

    @staticmethod
    def _get_time(model, sequence_subset=None, **kwargs):
        """Get time in days associated with states timesteps relative to model start date."""
        _ = kwargs
        dates = model.result_dates
        sec_in_day = 86400
        t = (dates - model.start).total_seconds().values / sec_in_day
        return t if sequence_subset is None else t[sequence_subset]

    def _get_connection_factors(self, model, sequence_subset=None, **kwargs):
        # FIXME calls the field's method twice
        _ = kwargs
        if sequence_subset is not None:
            res_dates = model.result_dates
            if res_dates.size:
                res_dates = res_dates[sequence_subset]
            date_range = (res_dates[0], res_dates[-1])
        else:
            date_range = None
        if hasnested(self.sample_attrs, 'MASKS', 'CF_MASK'):
            return model.get_spatial_connection_factors_and_perforation_ratio(date_range=date_range)[0]
        return None

    def _get_perforation_mask(self, model, sequence_subset=None, **kwargs):
        # FIXME calls the field's method twice
        _ = kwargs
        if sequence_subset is not None:
            res_dates = model.result_dates
            if res_dates.size:
                res_dates = res_dates[sequence_subset]
            date_range = (res_dates[0], res_dates[-1])
        else:
            date_range = None
        if hasnested(self.sample_attrs, 'MASKS', 'PERF_MASK'):
            return model.get_spatial_connection_factors_and_perforation_ratio(date_range=date_range)[1]
        return None

    @staticmethod
    def to_dates(model, t):
        """Restore actual dates from time deltas."""
        dates = model.start + np.array([pd.Timedelta(i, unit='day') for i in t])
        return pd.to_datetime(dates)

    @staticmethod
    def _get_distances(model, fill_invalid_neighbours=INVALID_VALUE_FILLER, neighbouring_radius=-1, **kwargs):
        """Get matrix of distances for neighbouring cells."""
        _ = kwargs
        return model.grid.calculate_neighbours_distances(
            connectivity=neighbouring_radius,
            fill_value=fill_invalid_neighbours
        )

    @staticmethod
    def _get_xyz(model, **kwargs):
        _ = kwargs
        return model.grid.xyz

    def _get_states(self, model, attrs, sequence_subset=None, **kwargs):
        """Get stacked states sequence."""
        _ = kwargs
        if (self.preloaded is not None) and (sequence_subset is not None):
            return np.stack([getattr(model.states, attr)[sequence_subset] for attr in attrs], axis=1)
        return np.stack([getattr(model.states, attr) for attr in attrs], axis=1)

    @staticmethod
    def _get_rock(model, attrs, **kwargs):
        """Get stacked rock attributes."""
        _ = kwargs
        return np.stack([getattr(model.rock, attr) for attr in attrs], axis=0)

    @staticmethod
    def _get_tables(model, attrs):
        """Get sample table data"""
        return {
            attr: getattr(model.tables, attr).to_numpy() if attr in TABLES_WITHOUT_INDEX
            else getattr(model.tables, attr).to_numpy(include_index=True) for attr in attrs
        }

    @staticmethod
    def _get_control(model, attrs, sequence_subset=None, **kwargs):
        """Get control in a spatial form (defined for all cells, meaningful values in
        perforated cells, other cells are filled with zeros) with corresponding dates.
        """
        _ = kwargs
        if sequence_subset is not None:
            res_dates = model.result_dates
            if res_dates.size:
                res_dates = res_dates[sequence_subset]
            date_range = (res_dates[0], res_dates[-1])
        else:
            date_range = None
        filtered_attrs = attrs.copy()
        if 'PROD_PERF_MASK' in attrs:
            filtered_attrs.remove('PROD_PERF_MASK')
        if 'INJE_PERF_MASK' in attrs:
            filtered_attrs.remove('INJE_PERF_MASK')

        output = model.get_spatial_well_control(filtered_attrs, date_range=date_range, fill_shut=0., fill_outside=0.)
        if 'PROD_PERF_MASK' in attrs or 'INJE_PERF_MASK' in attrs:
            control = []
            i = 0
            for attr in attrs:
                if attr == 'PROD_PERF_MASK':
                    control.append(get_spatial_perf(model, sequence_subset, mode='PROD'))
                elif attr == 'INJE_PERF_MASK':
                    control.append(get_spatial_perf(model, sequence_subset, mode='INJE'))
                else:
                    control.append(output['control'][:, i][:, None])
                    i += 1
            output['control'] = np.concatenate(control, axis=1)
        return output

    def set_transform(self, transform):
        """Set transforms to be applied to each sample

        Parameters
        ----------
        transform: class
            Class of transform to apply
            list of Classes can be used to compose several transforms
        Returns
        -------
        out: FieldDataset
        """
        if not isinstance(transform, (list, tuple)):
            transform = [transform]
        self.transform = []
        for t in transform:
            if inspect.isclass(t) and issubclass(t, Transform):
                if issubclass(t, Normalize):
                    if self.std is None or self.mean is None:
                        raise RuntimeError("Dataset's statistics are not calculated!")
                    self.transform.append(t(
                        mean=self.filtered_statistics['MEAN'],
                        std=self.filtered_statistics['STD'],
                        unravel_model=self.unravel_model
                    ))
                else:
                    self.transform.append(t())
            else:
                self.transform.append(t)
        self.transform = Compose(self.transform)
        return self

    def dump_samples(self, path, n_epoch=1, prefix=None, state=True, **kwargs):
        """Dump samples from the dataset.

        Parameters
        ----------
        path: str
            Path to the directory for dump.
        n_epoch: int
            Number of times to pass through the dataset.
        prefix: str, None
            Prefix for dumped samples.
        state: bool
            If True, dump the state of the samples
        kwargs: dict
            Additional named arguments for sample.dump

        Returns
        -------

        """
        if not os.path.isdir(path):
            os.mkdir(path)
        prefix = prefix + '_' if prefix is not None else ''
        i = 0
        for _ in range(n_epoch):
            for sample in self:
                sample.dump(os.path.join(path, prefix+str(i)+'.hdf5'), state=state, **kwargs)
                i += 1
        return self

    def convert_to_other_fmt(self, new_root_dir, new_fmt='hdf5', results_to_events=True, **kwargs):
        """Convert dataset to a new format.

        Parameters
        ----------
        new_root_dir: str
            Directory to save converted dataset
        new_fmt: str
            Extension to use
        kwargs: dict
            Any additional named arguments passed to Field.dump

        Returns
        -------
        FieldDataset
        """
        if not os.path.exists(new_root_dir):
            os.makedirs(new_root_dir)
        for i, path in enumerate(self.files):
            path, _ = os.path.splitext(path)
            if os.path.split(path)[0]:
                os.makedirs(os.path.join(new_root_dir, os.path.split(path)[0]))
            path = '.'.join([path, new_fmt])

            model = self._load_model(i, force_wells_calculations=True)
            if results_to_events:
                model.wells.results_to_events(grid=model.grid)
            config = None if new_fmt == 'hdf5' else self.config
            model.dump(path=os.path.join(new_root_dir, path), config=config, **kwargs)

        self.__init__(
            src=new_root_dir,
            sample_attrs=self.sample_attrs,
            fmt=(new_fmt, ),
            subset_generator=self.subset_generator
        )
        return self

    @property
    def filtered_statistics(self):
        """Filters out non-normalized attrs and attrs, which are not presented in `sample_attrs`, from statistics."""
        filtered_stats = dict()
        for key, value in zip(('MEAN', 'STD', 'MIN', 'MAX'), (self.mean, self.std, self.min, self.max)):
            if value is None:
                raise RuntimeError("Dataset's statistics are not calculated!")
            filtered_stat = {
                comp: {} for comp in self.sample_attrs
                if comp not in NON_NORMALIZED_ATTRS and len(self.sample_attrs[comp]) > 0
            }
            for comp in filtered_stat:
                if comp not in value:
                    raise ValueError('Component "%s" is not presented in calculated statistics.' % comp)
                for attr in self.sample_attrs[comp]:
                    if attr not in value[comp]:
                        raise ValueError('Attribute "%s" of component "%s" is not presented in calculated statistics.'
                                         % (attr, comp))
                    filtered_stat[comp][attr] = value[comp][attr]
                if comp not in self._attrs_sampled_as_dict:
                    filtered_stat[comp] = np.stack(
                        [filtered_stat[comp][attr] for attr in self.sample_attrs[comp]]
                    )
            filtered_stats[key] = filtered_stat
        return filtered_stats

    def calculate_statistics(self):  # pylint: disable=too-many-branches
        """Calculate mean and std values for the attributes of the dataset."""
        # Change sampling behavior for statistics' calculation.
        subset_generator, self.subset_generator = self.subset_generator, None

        mean, mean_of_squares, std, minim, maxim = {}, {}, {}, {}, {}
        for comp in self.sample_attrs:
            if comp not in NON_NORMALIZED_ATTRS:
                mean[comp] = {attr: [] for attr in self.sample_attrs[comp]}
                mean_of_squares[comp] = {attr: [] for attr in self.sample_attrs[comp]}
                std[comp] = {attr: [] for attr in self.sample_attrs[comp]}
                minim[comp] = {attr: [] for attr in self.sample_attrs[comp]}
                maxim[comp] = {attr: [] for attr in self.sample_attrs[comp]}

        for i in range(len(self)):
            m, m_sq, mn, mx = self._get_model_statistics(i)
            for comp in m:#pylint:disable=consider-using-dict-items
                for attr in m[comp]:
                    mean[comp][attr].append(m[comp][attr])
                    mean_of_squares[comp][attr].append(m_sq[comp][attr])
                    minim[comp][attr].append(mn[comp][attr])
                    maxim[comp][attr].append(mx[comp][attr])

        for comp in mean:#pylint:disable=consider-using-dict-items
            for attr in mean[comp]:
                mean[comp][attr] = np.mean(mean[comp][attr], axis=0)
                mean_of_squares[comp][attr] = np.mean(mean_of_squares[comp][attr], axis=0)
                std[comp][attr] = np.sqrt(np.abs(mean_of_squares[comp][attr] - mean[comp][attr]**2))
                minim[comp][attr] = np.min(minim[comp][attr], axis=0)
                maxim[comp][attr] = np.max(maxim[comp][attr], axis=0)

        # Recover old sampling behavior
        self.subset_generator = subset_generator

        self.mean, self.std, self.min, self.max = mean, std, minim, maxim
        return self

    def _get_model_statistics(self, idx):
        """Get mean and mean of squares for the attributes of the model."""
        sample = self._get_sample(idx)
        mean, mean_of_squares, minim, maxim = {}, {}, {}, {}
        for comp in sample.keys():
            if comp.upper() in NON_NORMALIZED_ATTRS:
                continue
            mask = sample.masks.well_mask if comp.upper() == 'CONTROL' else sample.masks.actnum

            mean[comp] = dict()
            mean_of_squares[comp] = dict()
            minim[comp] = dict()
            maxim[comp] = dict()
            if comp in self._attrs_sampled_as_dict:
                ax = 0
                for attr, arr in sample[comp].items():
                    if attr.upper() == 'DISTANCES':
                        arr = arr.copy().astype(np.float)
                        arr[sample.masks.invalid_neighbours_mask] = np.nan
                    mean[comp][attr] = np.nanmean(arr, axis=ax)
                    mean_of_squares[comp][attr] = np.nanmean(np.power(arr, 2), axis=ax)
                    minim[comp][attr] = np.nanmin(arr, axis=ax)
                    maxim[comp][attr] = np.nanmax(arr, axis=ax)
            else:
                ax = 1 if comp not in SEQUENTIAL_ATTRS else (0, 2)
                if mask is not None:
                    arr = sample[comp][..., mask]
                else:
                    arr = sample[comp]

                comp_mean = overflow_safe_mean(arr, axis=ax)
                comp_mean_of_squares = overflow_safe_mean(np.power(arr, 2), axis=ax)
                comp_min = np.min(arr, axis=ax)
                comp_max = np.max(arr, axis=ax)
                for i, attr in enumerate(self.sample_attrs[comp]):
                    mean[comp][attr] = comp_mean[i]
                    mean_of_squares[comp][attr] = comp_mean_of_squares[i]
                    minim[comp][attr] = comp_min[i]
                    maxim[comp][attr] = comp_max[i]

        return mean, mean_of_squares, minim, maxim

    def dump_statistics(self, path):
        """Dump mean and std values of the dataset into a file."""
        if self.std is None or self.mean is None or self.min is None or self.max is None:
            raise RuntimeError("Dataset's statistics are not calculated!")
        with open(path, 'wb') as f:
            pickle.dump([self.mean, self.std, self.min, self.max], f)

    def load_statistics(self, path):
        """Load mean and std values of the dataset from a file."""
        with open(path, 'rb') as f:
            self.mean, self.std, self.min, self.max = pickle.load(f)
        for kind in ('mean', 'std', 'min', 'max'):
            stats = getattr(self, kind)
            upper_stats = {}
            for comp, value in stats.items():
                if isinstance(value, dict):
                    upper_stats[comp.upper()] = {}
                    for attr, arr in value.items():
                        upper_stats[comp.upper()][attr.upper()] = arr
                else:
                    upper_stats[comp.upper()] = value
            setattr(self, kind, upper_stats)
        if 'CONTROL' in self.sample_attrs and 'CONTROL' in self.mean:
            for kind in ('mean', 'std', 'min', 'max'):
                stats = getattr(self, kind)
                for k in stats['CONTROL']:
                    if k in CONTROL_TO_RESULTS_KW and CONTROL_TO_RESULTS_KW[k] in self.sample_attrs['CONTROL']:
                        stats['CONTROL'][CONTROL_TO_RESULTS_KW[k]] = stats['CONTROL'].pop(k)

    @property
    def sample_attrs(self):
        """Attributes represented in the samples."""
        return self._sample_attrs

    @sample_attrs.setter
    def sample_attrs(self, x):
        self._sample_attrs = {
            comp.upper(): [attr.upper() for attr in x[comp]] for comp in x
        }


class FieldSample(BaseComponent):
    """Class representing the samples from the dataset.


    Parameters
    ----------
    path: str, optional
        Path to the file. Only HDF5 files are supported at the moment.
    field: Field, optional
    dataset: FieldDataset, optional
    state: dict, optional
    sample: dict-like, optional

    """
    class _decorators:
        """Decorators for the FieldSample."""
        @classmethod
        def without_batch_dimension(cls, method):
            """Decorates sample methods to be applied without the batch dimension."""
            def decorated(instance, inplace=False, **kwargs):
                batch_dimension = instance.state.batch_dimension if hasattr(instance.state,
                                                                            'batch_dimension') else False
                if batch_dimension:
                    instance = instance.transformed(RemoveBatchDimension, inplace=inplace)
                    inplace = True
                instance = method(instance, inplace=inplace, **kwargs)
                if batch_dimension:
                    instance = instance.transformed(AddBatchDimension, inplace=inplace)
                return instance
            return decorated

    def __init__(self, path=None, field=None, dataset=None, state=None, **sample):
        super().__init__(**sample)
        self._path = path
        self._field = field
        self.sample_attrs = dataset.sample_attrs if dataset is not None else None
        self.dataset = dataset
        if state is not None:
            self.init_state(**state)

    def _nested_dicts_to_base_components(self, class_name, d):
        if isinstance(d, dict):
            d = BaseComponent(class_name=class_name, **d)
            for key, value in d.items():
                value = self._nested_dicts_to_base_components(key, value)
                setattr(d, key, value)
        return d

    def __setattr__(self, key, value):
        if key[0] != '_':
            value = self._nested_dicts_to_base_components(key.upper(), value)
        super().__setattr__(key, value)

    def empty_like(self):
        """Get an empty sample with the same state and the structure of embedded BaseComponents (if any)."""
        empty = super().empty_like()
        empty = FieldSample(field=self.field, dataset=self.dataset, state=empty.state.as_dict(), **dict(empty))
        empty.sample_attrs = self.sample_attrs
        return empty

    def copy(self):
        """Get a copy of the sample."""
        copy = super().copy()
        copy.dataset = self.dataset
        copy.field = self.field
        copy.sample_attrs = self.sample_attrs
        return copy

    def dump(self, path, **kwargs):
        """Dump the sample into a file.

        Parameters
        ----------
        path: str
            Path to the file.
        kwargs: dict
            Additional named arguments passed to BaseComponent's dump method.

        """
        fname = os.path.basename(path)
        fmt = os.path.splitext(fname)[1].strip('.')

        if fmt.upper() == 'HDF5':
            if hasattr(self.state, 'tensor') and self.state.tensor:
                out = self.transformed(ToNumpy)
                return out.dump(path, **kwargs)
            for state, value in self.state.as_dict().items():
                if issubclass(value.__class__, BaseComponent):
                    if state == 'sample_attributes':
                        for k, v in value.items():
                            value[k] = np.array(v, dtype='S16')
                    setattr(self, state, value)
                    self.set_state(**{state: 'base_component'})
            return self._dump_hdf5(path, **kwargs)
        raise NotImplementedError('File format {} not supported.'.format(fmt))

    def load(self, **kwargs):
        """Load sample from a file.

        Parameters
        ----------
        kwargs: dict
            Additional named arguments passed to the load method.

        Returns
        -------
        sample: FieldSample
            Sample with loaded data.
        """
        if self._path is None:
            raise RuntimeError('You should specify a path before loading!')
        fname = os.path.basename(self._path)
        fmt = os.path.splitext(fname)[1].strip('.')

        if fmt.upper() == 'HDF5':
            self._load_hdf5(self._path, **kwargs)
        else:
            raise NotImplementedError('File format {} not supported.'.format(fmt))
        for state, value in self.state.as_dict().items():
            if value == 'base_component':
                value = getattr(self, state)
                if state == 'sample_attributes':
                    for k, v in value.items():
                        value[k] = list(v.astype('U'))
                self.set_state(**{state: value})
                delattr(self, state)
        return self


    @property
    def field(self):
        """Link to the parent field."""
        return self._field

    @field.setter
    def field(self, x):
        if x is not None and not isinstance(x, Field):
            raise ValueError('Can assign only instances of the class %s!' % str(Field))
        self._field = x

    @property
    def dataset(self):
        """Link to the parent dataset."""
        return self._dataset

    @dataset.setter
    def dataset(self, x):
        if x is not None and not isinstance(x, FieldDataset):
            raise ValueError('Can assign only instances of the class %s!\nGiven %s' % (str(FieldDataset), type(x)))
        self._dataset = x
        if x is not None:
            self.init_state(
                spatial=x.unravel_model,
                cropped_at_mask=None if x.unravel_model else 'ACTNUM'
            )
            try:
                self.init_state(dataset_statistics=self._nested_dicts_to_base_components(
                    'DATASET_STATISTICS', x.filtered_statistics
                ))
            except RuntimeError:
                pass

    def transformed(self, transforms, inplace=False):
        """Apply a set of transforms to the sample.

        Parameters
        ----------
        transforms: list, tuple, Compose, Transform
            Transform to apply
        inplace: bool

        Returns
        -------
        sample: FieldSample
            Transformed sample.
        """
        transforms = self._initialize_transform(transforms)
        return transforms(self, inplace=inplace)

    def at_wells(self, inplace=False):
        """Crop all the spatial arrays to the perforated cells. Ravel if needed.

        Parameters
        ----------
        inplace: bool

        Returns
        -------
        sample: FieldSample
            Cropped sample.
        """
        return self.as_ravel(inplace=inplace, crop_at_mask='WELL_MASK')

    @_decorators.without_batch_dimension
    def as_spatial(self, inplace=False):
        """Transform the sample's arrays to the spatial form.

        Parameters
        ----------
        inplace: bool

        Returns
        -------
        sample: FieldSample
        """
        raise NotImplementedError()

    # pylint: disable=too-many-nested-blocks
    @_decorators.without_batch_dimension
    def as_ravel(self, inplace=False, crop_at_mask='ACTNUM'):
        """Ravel the sample's arrays.

        Parameters
        ----------
        inplace: bool

        Returns
        -------
        sample: FieldSample
        """
        out = self if inplace else self.empty_like()
        if self.state.spatial:
            for comp in self.keys():#pylint:disable=consider-using-dict-items
                if comp.upper() == 'TABLES':
                    out[comp] = self[comp]
                    continue
                if comp.upper() in ('MASKS', 'GRID'):
                    for attr in self[comp].keys():
                        if attr.upper() in ('TIME', 'CONTROL_T'):
                            out[comp][attr] = self[comp][attr]
                            continue
                        if attr.upper() == 'NAMED_WELL_MASK':
                            for well in self[comp][attr].keys():
                                new_shape = (-1,) + tuple(self[comp][attr][well].shape[3:])
                                out[comp][attr][well] = \
                                    self[comp][attr].reshape(attr=well, newshape=new_shape, order='F', inplace=False)
                            continue
                        if attr.upper() in ('CF_MASK', 'PERF_MASK'):
                            new_shape = tuple(self[comp][attr].shape[:-3]) + (-1,)
                        else:
                            new_shape = (-1,) + tuple(self[comp][attr].shape[3:])
                        out[comp][attr] = self[comp].reshape(attr=attr, newshape=new_shape, order='F', inplace=False)
                else:
                    new_shape = tuple(self[comp].shape[:-3]) + (-1, )
                    out[comp] = self.reshape(attr=comp, newshape=new_shape, order='F', inplace=False)
        out.set_state(spatial=False)
        if crop_at_mask != self.state.cropped_at_mask:
            if self.state.cropped_at_mask is not None:
                out = self._uncrop_from_mask(out, self.state.cropped_at_mask)
            out = self._crop_at_mask(out, crop_at_mask)
        return out

    @staticmethod
    def _crop_at_mask(obj, mask_name):
        """Crop a sample at a given binary mask.

        Parameters
        ----------
        obj: FieldSample
            Sample to be cropped.
        mask_name: str
            Name of the mask from the sample['MASKS'].

        Returns
        -------
        obj: FieldSample
            Cropped sample.
        """
        assert not obj.state.spatial
        mask = obj.masks[mask_name]
        if isinstance(mask, torch.Tensor):
            mask = mask.bool()
        else:
            mask = mask.astype(bool)
        for comp in obj.keys():
            if comp.upper() == 'TABLES':
                continue
            if comp.upper() in ('MASKS', 'GRID'):
                for attr in obj[comp].keys():
                    if attr.upper() in ('CF_MASK', 'PERF_MASK'):
                        obj[comp][attr] = obj[comp][attr][..., mask]
                    elif attr.upper() == 'NAMED_WELL_MASK':
                        for well in obj[comp][attr].keys():
                            obj[comp][attr][well] = obj[comp][attr][well][..., mask]
                    elif attr.upper() not in (mask_name.upper(), 'TIME', 'CONTROL_T') and obj[comp][attr] is not None:
                        obj[comp][attr] = obj[comp][attr][mask]
            else:
                obj[comp] = obj[comp][..., mask]
        obj.set_state(cropped_at_mask=mask_name.upper())
        return obj

    @staticmethod
    def _uncrop_from_mask(obj, mask_name):
        """Reverse operation to the crop_at_mask.

        Parameters
        ----------
        obj: FieldSample
        mask_name: str

        Returns
        -------
        obj: FieldSample
        """
        raise NotImplementedError()

    def _initialize_transform(self, transforms):
        """Initialize transforms before application."""
        if not isinstance(transforms, (list, tuple, Compose)):
            transforms = [transforms]
        initialized_transforms = []
        for t in transforms:
            if inspect.isclass(t):
                if issubclass(t, Normalize):
                    initialized_transforms.append(t(
                        mean=self.state.dataset_statistics.mean,
                        std=self.state.dataset_statistics.std,
                        unravel_model=self.state.spatial
                    ))
                else:
                    initialized_transforms.append(t())
            else:
                initialized_transforms.append(t)
        return Compose(initialized_transforms)

    @property
    def sample_attrs(self):
        """Attributes represented in the sample."""
        return self.state.sample_attributes

    @sample_attrs.setter
    def sample_attrs(self, x):
        x = None if x is None else {comp.upper(): [attr.upper() for attr in x[comp]] for comp in x.keys()}
        x = self._nested_dicts_to_base_components('SAMPLE_ATTRIBUTES', x)
        if hasattr(self.state, 'SAMPLE_ATTRIBUTES'):
            self.set_state(sample_attributes=x)
        else:
            self.init_state(sample_attributes=x)

    @property
    def device(self):
        """Get the sample's device (if it is in Torch format)

        Returns
        -------
        device: torch.device
        """
        ref = None
        for _, value in self.items():
            if isinstance(value, BaseComponent):
                for _, arr in value.items():
                    if isinstance(arr, BaseComponent):
                        for _, mask in arr.items():
                            ref = mask
                            break
                    else:
                        ref = arr
                    break
            else:
                ref = value
        if ref is None:
            raise RuntimeError('The sample is empty!')
        if isinstance(ref, torch.Tensor):
            return ref.device
        raise RuntimeError('The sample should be in the PyTorch format! Found: %s' % type(ref))

    def to(self, device, inplace=True):
        """Change the sample's device (if it is in Torch format).

        Parameters
        ----------
        device: str, torch.device
        inplace: bool

        Returns
        -------
        sample: FieldSample
            Sample at the new device
        """
        if self.device == device:
            return self if inplace else self.copy()
        out = self if inplace else self.empty_like()
        for comp, value in self.items():
            if isinstance(value, BaseComponent):
                for attr, arr in value.items():
                    if isinstance(arr, BaseComponent):
                        for well, mask in arr.items():
                            out[comp][attr][well] = mask.to(device)
                    else:
                        out[comp][attr] = arr.to(device)
            else:
                out[comp] = value.to(device)
        return out


class SequenceSubset:
    """Baseclass for generating subsets of sequences."""
    def __init__(self, size, low, high, **kwargs):
        """
        Parameters
        ----------
        size: int
            Lenght of the generated sequences
        low: int
            Minimal possible timestep (inclusive)
        high: int
            Maximal possible timestep (exclusive)
        kwargs: optional
        """
        self.size = size
        self.low = low
        self.high = high
        _ = kwargs

    def __call__(self):
        """Generate subset of timesteps."""
        raise NotImplementedError('Abstract method is not implemented.')


class UniformSequenceSubset(SequenceSubset):
    """Generator of timesteps sampled from a uniform distribution [low, high)."""
    def __call__(self):
        subset = np.random.choice(np.arange(self.low, self.high), size=self.size, replace=False)
        return np.sort(subset)


class RandomSubsequence(SequenceSubset):
    """Generator of timestep subsequences."""
    def __call__(self):
        start = np.random.randint(low=self.low, high=self.high - self.size + 1)
        subset = np.arange(start, start + self.size)
        return np.sort(subset)
