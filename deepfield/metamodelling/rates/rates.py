"""Module for differentiable rates calculation."""
from copy import copy

import torch
from torch import nn
from .table_interpolation import get_callable_table, get_baker_linear_model
from ...datasets.transforms import Denormalize, RemoveBatchDimension


class RatesModule(nn.Module):
    """Differentiable rates calculation."""

    minimal_sample_attrs = {
        'STATES': ['PRESSURE', 'SOIL', 'SWAT', 'SGAS', 'RS'],
        'ROCK': ['PORO', 'PERMX', 'PERMY', 'PERMZ'],
        'CONTROL': ['BHPT'],
        'TABLES': ['PVTO', 'PVTW', 'PVDG', 'SWOF', 'SGOF', 'DENSITY'],
        'MASKS': ['TIME'],
        'GRID': ['XYZ'],
        'RATES': []
    }

    allowed_rates_attrs = ('BOPR', 'BWPR', 'BGPR')
    rates_to_pvt_name = {'BGPR': 'PVDG', 'BOPR': 'PVTO', 'BWPR': 'PVTW'}
    rates_to_pvt_attrs = {'BGPR': ('PRESSURE', ), 'BOPR': ('RS', 'PRESSURE'), 'BWPR': ('PRESSURE', )}
    rates_to_rp_name = {'BGPR': 'SGOF', 'BOPR': 'BAKER', 'BWPR': 'SWOF'}
    rates_to_rp_attrs = {'BGPR': ('SGAS', ), 'BOPR': ('SWAT', 'SGAS'), 'BWPR': ('SWAT', )}

    def __init__(self, units='METRIC'):
        super().__init__()
        self.g = 0.0000980665 if units == 'METRIC' else 0.00694

    def forward(self, sample, inplace=False, use_gravitational_adjustment=False, bhp_at_upper_perf=True):
        """Calculate rates for a given sample.

        Parameters
        ----------
        sample: FieldSample
            Contains attributes of a Field requested via `sample_attrs'.
            NOTE:
                Here and further we will denote by N a maximal number of perforated cells
                at the time-interval covered by the sample.

            EXAMPLE:

                sample = {

                    'states': torch.Tensor of shape [n_ts, n_states_ch, N],

                    'rock': torch.Tensor of shape [n_rock_ch, N],

                    'control': torch.Tensor of shape [n_ts - 1, n_control_ch, N],

                    'tables': {
                        'pvto': torch.Tensor of shape [n_pvto_rows, n_pvto_columns],
                            ...,
                        'table_name': torch.Tensor of shape [n_table_rows, n_table_columns],
                    }

                    'masks': {
                        'time': torch.Tensor of shape [n_ts],
                        'perf_mask': torch.Tensor of shape [n_ts, N],
                        'xyz': torch.Tensor of shape [8, 3, N]
                    }

                }

            NOTE:
                sample['masks']['perf_mask'][t] indicates which cells are perforated at timestep t.
                Mask contains ones in entries corresponding to currently perforated cells, and zeros else.
                If 'perf_mask' is not specified, all cells are assumed to be always perforated.

        sample_attrs: dict
            Dict containing names and order of attributes presented in `sample`.
            Example:

            sample_attrs = {
                'states': ['pressure', 'soil', 'swat', 'sgas', 'rs'],
                'rock': ['poro', 'permx', 'permy', 'permz'],
                'control': ['bhpt'],
                'tables': ['pvto', 'pvtw', 'pvdg', 'swof', 'sgof', 'density'],
                'masks': ['time', 'perf_mask', 'xyz'],
                'rates': ['bopr', 'bwpr', 'bgpr']
            }

            NOTE:
                `sample_attrs` contains key 'rates', which indicates what kind of rates should be computed!

        Returns
        -------
        rates: torch.Tensor
            Computed rates, based on attributes in `sample_attrs['rates']`.
            Shape: [n_ts - 1, len(sample_attrs['rates']), N]

        """
        sample_attrs = sample.sample_attrs
        if 'RATES' not in sample_attrs:
            sample_attrs['RATES'] = ['BOPR', 'BWPR', 'BGPR']
            if inplace:
                sample.sample_attrs = sample_attrs
        sample_unchanged = sample

        batch_dimension = hasattr(sample.state, 'batch_dimension') and sample.state.batch_dimension
        if batch_dimension:
            sample = sample.transformed(RemoveBatchDimension)
        if sample.state.cropped_at_mask != 'WELL_MASK':
            sample = sample.at_wells()
        if hasattr(sample.state, 'normalized') and sample.state.normalized:
            sample = sample.transformed(Denormalize)

        states_to_ind = {attr: i for i, attr in enumerate(sample_attrs['STATES'])}
        control_to_ind = {attr: i for i, attr in enumerate(sample_attrs['CONTROL'])}

        states = {attr: sample['STATES'][1:, states_to_ind[attr]] for attr in sample_attrs['STATES']}
        control = {attr: sample['CONTROL'][:, control_to_ind[attr]] for attr in sample_attrs['CONTROL']}
        control['BHPT'][control['BHPT'] == 0] = states['PRESSURE'][control['BHPT'] == 0]
        tables = {attr: get_callable_table(attr, sample['TABLES'][attr]) for attr in sample_attrs['TABLES']}
        tables['BAKER'] = get_baker_linear_model(sample['TABLES']['SWOF'], sample['TABLES']['SGOF'])
        perf = sample['MASKS']['PERF_MASK'].squeeze(1)
        cf = sample['MASKS']['CF_MASK'].squeeze(1) * perf
        depression = states['PRESSURE'] - control['BHPT']

        rates, fvf, mu, kr = {}, {}, {}, {}

        if use_gravitational_adjustment:
            h = (sample['GRID']['XYZ'][:, 4:, 2] - sample['GRID']['XYZ'][:, :4, 2]).mean(dim=1)
            rho = sample['TABLES']['DENSITY'][0]
            rho = {attr: rho[i] for i, attr in enumerate(self.allowed_rates_attrs)}

        rates_to_compute = copy(sample_attrs['RATES'])
        if 'BGPR' in rates_to_compute and 'BOPR' not in rates_to_compute:
            rates_to_compute.append('BOPR')

        for attr in rates_to_compute:
            pvt_name = self.rates_to_pvt_name[attr]
            pvt_attrs = torch.stack(tuple(states[s] for s in self.rates_to_pvt_attrs[attr]), dim=0)
            fvf[attr], mu[attr] = tables[pvt_name](pvt_attrs, columns_dim=0)

            rp_name = self.rates_to_rp_name[attr]
            rp_attrs = torch.stack(tuple(states[s] for s in self.rates_to_rp_attrs[attr]), dim=0)
            kr[attr] = tables[rp_name](rp_attrs, columns_dim=0)[0]

            if use_gravitational_adjustment:
                sign = -1 if bhp_at_upper_perf else 1
                dp = depression + sign * (rho[attr] * fvf[attr] * self.g * h)
            else:
                dp = depression

            rates[attr] = dp * cf * kr[attr] / (mu[attr] * fvf[attr])
            rates[attr][perf == 0] = 0

        if 'BGPR' in rates_to_compute:
            rates['BGPR'] += states['RS'] * rates['BOPR']

        rates = torch.stack([rates[attr] for attr in sample_attrs['RATES']], dim=1)
        if batch_dimension:
            rates = rates.unsqueeze(0)
        if inplace:
            sample_unchanged.rates = rates
        return rates

    def check_if_minimal_sample_attrs_satisfied(self, sample_attrs):
        """Checks if minimal sample_attrs requirements are satisfied.

        Parameters
        ----------
        sample_attrs: dict

        """
        for comp, attrs in self.minimal_sample_attrs.items():
            if comp in sample_attrs:
                for attr in attrs:
                    if attr not in sample_attrs[comp]:
                        raise ValueError('Attr "%s" is not presented in given `sample_attrs["%s"]`!' % (attr, comp))
            else:
                raise ValueError('Key "%s" is not presented in given `sample_attrs`!' % comp)
        if 'rates' in sample_attrs:
            for attr in sample_attrs['rates']:
                if attr not in self.allowed_rates_attrs:
                    raise ValueError(
                        'Rates type "%s" is not understood. Choose rate types from the list: %s'
                        % (attr, self.allowed_rates_attrs)
                    )
        else:
            raise ValueError('You should specify rates in sample_attrs.')
