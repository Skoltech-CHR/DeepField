"""Tools for generating randomized datasets."""
import os
import numpy as np
from scipy.ndimage import gaussian_filter

from ..field import Field, Rock, States, Wells

from .utils import get_config, STATES_KEYWORD


def _apply_uncorrelated_noise(arr, std):
    noise = np.random.randn(*arr.shape) * std
    return arr + noise


def _apply_correlated_noise(arr, std, freq):
    seeds = np.random.choice(
        [0, 1], size=arr.shape,
        p=[1 - freq, freq]
    )
    std = std / freq * seeds

    noise = np.random.randn(*arr.shape) * std
    noise = gaussian_filter(noise, 1 / freq)
    return arr + noise


def _get_uniform_samples(*args):
    if len(args) > 1:
        return tuple(np.random.rand() * (arg[1] - arg[0]) + arg[0] for arg in args)
    return np.random.rand() * (args[0][1] - args[0][0]) + args[0][0]


def _noise_component(size, r):
    out = np.random.rand(size) * (r[1] - r[0]) + r[0]
    return out


def _const_component(size, b):
    b = _get_uniform_samples(b)
    out = np.zeros(size) + b
    return out


def _exp_component(size, a, v):
    a, v = _get_uniform_samples(a, v)
    l = np.exp(v)
    t = np.arange(size)
    out = a * np.exp(-t * l)
    return out


def _sin_component(size, scale, phi, minima):
    scale, phi, minima = _get_uniform_samples(scale, phi, minima)
    t = np.arange(size)
    out = np.sin(scale * t + phi)
    out = (1 + out) * (1 - minima) / 2 + minima
    return out


class AttrRandomizer:
    """Baseclass for attribute randomization."""
    def __init__(self, associated_class, **kwargs):
        for key, arg in kwargs.items():
            setattr(self, key, arg)
        self.associated_class = associated_class

    def __call__(self, attr, **kwargs):
        """Get randomized version of attr."""
        if attr.__class__ != self.associated_class:
            raise ValueError('Can randomize instances of %s class. Found: %s' % (self.associated_class, attr.__class__))
        return self._randomize_attr(attr, **kwargs)

    def _randomize_attr(self, *args, **kwargs):
        raise NotImplementedError('Abstract method.')


class ControlRandomizer(AttrRandomizer):
    """Randomizer for control attributes (works inplace!)."""
    def __init__(self, attr_to_vary='BHPT', equality_condition=None, exp_amp=(70, 250), exp_log_curv=(-6, -4),
                 const=(1, 10), sin_scale=(0.001, 0.02), sin_phi=(0, 2 * np.pi), sin_minima=(0.6, 0.8),
                 noise_range=(0, 0)):
        """
        Parameters
        ----------
        attr_to_vary: str
            Attribute name to vary (should be represented in well's events)
        """

        kwargs = dict(attr_to_vary=attr_to_vary, equality_condition=equality_condition, exp_amp=exp_amp,
                      exp_log_curv=exp_log_curv, const=const, sin_scale=sin_scale, sin_phi=sin_phi,
                      sin_minima=sin_minima, noise_range=noise_range)
        super().__init__(associated_class=Wells, **kwargs)

    def _randomize_attr(self, wells, **kwargs):
        """
        Parameters
        ----------
        wells: deepfield.field.Wells

        Returns
        -------
        out: deepfield.field.Wells
            Wells with randomized events.
        """
        _ = kwargs

        attr = self.attr_to_vary
        exp_amp, exp_curv = self.exp_amp, self.exp_log_curv
        const = self.const
        sin_scale, sin_phi, sin_minima = self.sin_scale, self.sin_phi, self.sin_minima
        noise_range = self.noise_range

        def sampler(size):
            e = _exp_component(size, exp_amp, exp_curv)
            c = _const_component(size, const)
            s = _sin_component(size, sin_scale, sin_phi, sin_minima)
            eps = _noise_component(size, noise_range)
            out = e * s + c + eps
            return out

        clip_min = 0
        clip_max = None
        kwargs = {'additive': False, 'clip': (clip_min, clip_max),
                  'equality_condition': self.equality_condition, attr: sampler}
        wells.randomize_events(**kwargs)
        return wells


class StatesRandomizer(AttrRandomizer):
    """Randomizer for states attributes"""
    def __init__(self, std_reference_func=np.max, std_amplitude=0.01, uncorrelated_noise=False,
                 correlated_noise_freq=0.1, inplace=False):
        """
        Parameters
        ----------
        std_reference_func: callable
            Reference function used to calculate standard deviation of randomization
            Output of this function will be multiplied by std_amplitude and this value will be used as std.
        std_amplitude: float
            Multiplier for output of std_reference_func.
            The result of multiplication will be used as std of randomization.
        uncorrelated_noise: bool
            If False, correlated noise will be applied.
        inplace: bool
        """
        kwargs = dict(std_reference_func=std_reference_func, std_amplitude=std_amplitude,
                      uncorrelated_noise=uncorrelated_noise, inplace=inplace,
                      correlated_noise_freq=correlated_noise_freq, sat_attrs=('SOIL', 'SWAT', 'SGAS'))
        super().__init__(associated_class=States, **kwargs)

    def _randomize_attr(self, states, **kwargs):
        """
        Parameters
        ----------
        states: deepfield.field.States

        Returns
        -------
        out: deepfield.field.States
            Randomized states.
        """
        actnum = kwargs['actnum']
        noisy_states = self._apply_noise(states, actnum)

        if not self.inplace:
            states = States()
        for attr, arr in noisy_states.items():
            setattr(states, attr, arr)
        return states

    def _apply_noise(self, states, actnum):
        noisy_states = {}
        for attr in states.attributes:
            arr = getattr(states, attr)
            std = self.std_reference_func(arr) * self.std_amplitude

            if self.uncorrelated_noise:
                arr = _apply_uncorrelated_noise(arr, std)
            else:
                arr = _apply_correlated_noise(arr, std, self.correlated_noise_freq)

            clip_min = 1 if attr == 'PRESSURE' else 0
            clip_max = 1 if attr in self.sat_attrs else None
            arr = np.clip(arr, a_min=clip_min, a_max=clip_max)
            arr *= actnum
            noisy_states[attr] = arr
        noisy_states = self._normalize_saturations(noisy_states)
        return noisy_states

    def _normalize_saturations(self, states):
        sat_attrs = [attr for attr in self.sat_attrs if attr in states]
        if len(sat_attrs) > 1:
            sat_denominator = np.sum(
                [states[attr] for attr in sat_attrs],
                axis=0
            )
            for attr in sat_attrs:
                states[attr] = np.divide(states[attr], sat_denominator, out=np.zeros_like(states[attr]),
                                         where=sat_denominator != 0)
        return states


class RockRandomizer(AttrRandomizer):
    """Randomizer for rock attributes."""
    def __init__(self, std_reference_func=np.max, std_amplitude=0.01, uncorrelated_noise=False,
                 correlated_noise_freq=0.1, inplace=False):
        """
        Parameters
        ----------
        std_reference_func: callable
            Reference function used to calculate standard deviation of randomization
            Output of this function will be multiplied by std_amplitude and this value will be used as std.
        std_amplitude: float
            Multiplier for output of std_reference_func.
            The result of multiplication will be used as std of randomization.
        uncorrelated_noise: bool
            If False, correlated noise will be applied.
        inplace: bool
        """
        kwargs = dict(std_reference_func=std_reference_func, std_amplitude=std_amplitude,
                      uncorrelated_noise=uncorrelated_noise, inplace=inplace,
                      correlated_noise_freq=correlated_noise_freq)
        super().__init__(associated_class=Rock, **kwargs)

    def _randomize_attr(self, rock, **kwargs):
        """
        Parameters
        ----------
        rock: deepfield.field.Rock

        Returns
        -------
        out: deepfield.field.Rock
            Randomized rock.
        """
        actnum = kwargs['actnum']
        noisy_rock = self._apply_noise(rock, actnum)

        if not self.inplace:
            rock = Rock()
        for attr, arr in noisy_rock.items():
            setattr(rock, attr, arr)
        return rock

    def _apply_noise(self, rock, actnum):
        perm_attrs = [attr for attr in ('PERMX', 'PERMY', 'PERMZ') if hasattr(rock, attr)]
        if perm_attrs:
            perm = getattr(rock, perm_attrs[0])
            perm[perm == 0] = 1
            relative_perm = {attr: getattr(rock, attr) / perm for attr in perm_attrs[1:]}
        else:
            perm, relative_perm = {}, {}

        noisy_rock = {}
        for attr in rock.attributes:
            if attr in relative_perm:
                continue
            arr = getattr(rock, attr)
            std = self.std_reference_func(arr) * self.std_amplitude

            if self.uncorrelated_noise:
                arr = _apply_uncorrelated_noise(arr, std)
            else:
                arr = _apply_correlated_noise(arr, std, self.correlated_noise_freq)

            clip_min = 0
            clip_max = 1 if attr == 'PORO' else None
            arr = np.clip(arr, a_min=clip_min, a_max=clip_max)
            arr *= actnum
            noisy_rock[attr] = arr
        for attr, arr in relative_perm.items():
            noisy_rock[attr] = arr * noisy_rock[perm_attrs[0]]
        return noisy_rock


class FieldRandomizer:
    """Randomization of Fields, based on some base model."""
    default_states_rand = StatesRandomizer(std_reference_func=np.max, std_amplitude=0.01,
                                           uncorrelated_noise=True, inplace=True)
    default_rock_rand = RockRandomizer(std_reference_func=np.max, std_amplitude=0.01,
                                       uncorrelated_noise=True, inplace=True)
    default_control_rand = ControlRandomizer()

    default_randomizer = {
        'states': default_states_rand,
        'rock': default_rock_rand,
        'control': default_control_rand,
    }

    def __init__(self, path_to_base_field, randomizer=None):
        """
        Parameters
        ----------
        path_to_base_field: std
            Path to the base-field used to randomize around.
        randomizer: dict
            Dict with instances of randomizers for states, rock and control (wells)
            Should have keys the following form:
                randomizer = {
                    'states': states_rand,
                    'rock': rock_rand,
                    'wells': control_rand,
                }
        """
        self.config = get_config()
        self.config[STATES_KEYWORD] = {'attrs': self.config[STATES_KEYWORD]['attrs'], 'subset': [0]}

        self.base_path = path_to_base_field

        if randomizer is not None:
            self.randomizer = {}
            for key, def_r in self.default_randomizer.items():
                if key not in randomizer.keys() or randomizer[key] is None:
                    self.randomizer[key] = def_r
                else:
                    self.randomizer[key] = randomizer[key]
        else:
            self.randomizer = self.default_randomizer

    def _load_model(self, path):
        """Load and unravel field model."""
        model = Field(path=path, config=self.config, encoding='auto:10000', loglevel='ERROR')
        model.load(raise_errors=False)
        model.to_spatial()
        if 'wells' in map(lambda s: s.lower(), self.config.keys()):
            model.wells.drop_incomplete()
            model.wells.get_wellblocks(grid=model.grid)
            model.wells.apply_perforations()
            model.wells.add_null_results_column(column='WWIR')
            model.wells.results_to_events(grid=model.grid)
        return model

    def get_randomized_field(self):
        """Get randomized field.

        Returns
        -------
        out: deepfield.field.Field
            Field with randomized initial state, rock and control.
        """
        field = self._load_model(self.base_path)
        kwargs = {'actnum': field.grid.actnum.astype(np.float)}
        for key, rnd in self.randomizer.items():
            attr = key if key != 'control' else 'wells'
            if hasattr(field, attr):
                value = getattr(field, attr)
                randomizers = rnd if isinstance(rnd, tuple) else (rnd, )
                for randomizer in randomizers:
                    value = randomizer(value, **kwargs)
                setattr(field, attr, value)
        return field

    def generate_randomized_dataset(self, root_dir, n_samples, fmt=None, title=None):
        """Generate dataset consisting of randomized fields.

        Parameters
        ----------
        root_dir: str
            Path to the directory to create the dataset
        n_samples: int
            An amount of fields to generate
        fmt: str, optional
            Format in which fields will be dumped (with field.dump(fmt=fmt))
            If None, fields will be dumped in tNavigator format.
        title: str
        """
        if not os.path.exists(root_dir):
            os.mkdir(root_dir)
        fmt = ('.' + fmt) if fmt is not None else ''
        title = (title + '_') if title is not None else ''
        for i in range(n_samples):
            field = self.get_randomized_field()
            filename = title + str(i) + fmt
            path = os.path.join(root_dir, filename)
            if fmt == '':
                os.mkdir(path)
            field.dump(path=path, config=self.config, mode='w')
