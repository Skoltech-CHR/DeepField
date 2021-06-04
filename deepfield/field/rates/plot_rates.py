"""Methods to plot rates results"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from ipywidgets import interact
import ipywidgets as widgets

COLORS = {'Oil': 'black', 'Water': 'blue', 'Gas': 'orange', 'Free Gas': 'red'}
COMPARE_COLOR = 'green'
NAMES_TO_KW = {'Oil': 'WOPR', 'Water': 'WWPR', 'Gas': 'WGPR', 'Free Gas': 'WFGPR'}

def _safe_data_getter(df, time_start, time_end, kw):
    """Get columns data."""
    if df.empty:
        return np.full(time_end - time_start, np.nan)
    if kw in df:
        return df.loc[time_start:time_end, kw].values
    return np.full(len(df.loc[time_start:time_end]), np.nan)

def show_blocks_dynamics(wells, timesteps, wellnames, figsize=(16, 6)):
    """Plot liquid or gas rates and pvt props for a chosen block of a chosen well segment on two separate axes.

    Parameters
    ----------
    wells : Wells class instance
        Wells component with calculated rates and properties in blocks_dynamics attribute.
    timesteps : list of Timestamps
        Dates at which rates were calculated.
    wellnames : array-like
        List of active producing wells.
    figsize : tuple
        Figsize for two axes plots.
    """
    def update(wellname, block, rate, pvt_prop, time_start, time_end):
        dynamics = getattr(wells[wellname], 'blocks_dynamics', pd.DataFrame(dict(DATE=timesteps)))

        data = _safe_data_getter(dynamics, time_start, time_end, NAMES_TO_KW.get(rate, rate))
        data = np.array([np.stack(x) for x in data])
        prod_rate = data[..., block]

        data = _safe_data_getter(dynamics, time_start, time_end, NAMES_TO_KW.get(pvt_prop, pvt_prop))
        data = np.array([np.stack(x) for x in data])
        pvt = data[..., block]

        _, axes = plt.subplots(1, 2, figsize=figsize)

        axes[0].set_title(f'Well {wellname.upper()}. {rate}')
        axes[0].set_ylabel('Rate')
        axes[0].set_xlabel('Timestep')
        axes[0].plot(dynamics.loc[time_start:time_end, 'DATE'], prod_rate,
                     'o-', c=COLORS[rate], ms=3.5, lw=1.3)
        axes[0].axvline(dynamics.loc[time_start, 'DATE'], ls=':', c='grey')
        axes[0].axvline(dynamics.loc[time_end, 'DATE'], ls=':', c='grey')
        axes[0].grid(True)

        axes[1].set_title(f'Well {wellname.upper()}. {pvt_prop}')
        axes[1].set_ylabel('PVT Property')
        axes[1].set_xlabel('Timestep')
        axes[1].plot(dynamics.loc[time_start:time_end, 'DATE'], pvt,
                     'o-', c='black', ms=3.5, lw=1.3)
        axes[1].axvline(dynamics.loc[time_start, 'DATE'], ls=':', c='grey')
        axes[1].axvline(dynamics.loc[time_end, 'DATE'], ls=':', c='grey')
        axes[1].grid(True)

    well_ind_widget = widgets.Dropdown(options=wellnames)
    block_widget = widgets.Dropdown(options=[(tuple(item), i) for i, item in
                                             enumerate(wells[well_ind_widget.value].blocks)])

    timesteps_len = len(timesteps) - 1

    def update_block_list(*args):
        name = args[0]['new']
        block_widget.options = [(tuple(item), i) for i, item in enumerate(wells[name].blocks)]
    well_ind_widget.observe(update_block_list, 'value')

    rate_widget = widgets.Dropdown(options=['Oil', 'Water', 'Gas', 'Free Gas'], value='Oil')
    pvt_prop_widget = widgets.Dropdown(options=['WBHP', 'FVF_O', 'FVF_W', 'FVF_G', 'VISC_O', 'VISC_W',
                                                'VISC_G', 'KR_O', 'KR_W', 'KR_G'], value='WBHP')

    interact(update,
             wellname=well_ind_widget,
             block=block_widget,
             rate=rate_widget,
             pvt_prop=pvt_prop_widget,
             time_start=widgets.IntSlider(min=0, max=timesteps_len, step=1, value=0),
             time_end=widgets.IntSlider(min=0, max=timesteps_len, step=1, value=timesteps_len))
    plt.show()

def show_rates(wells, timesteps, wellnames, wells2=None, labels=None, figsize=(16, 6)):
    """Plot liquid and gas rates for a chosen well segment on two separate axes.

    Parameters
    ----------
    wells : Wells class instance
        Wells component with calculated rates in results attribute.
    timesteps : list of Timestamps
        Dates at which rates were calculated.
    wells2 : Wells class instance
        Wells component with results for comparison.
    labels : array-like
        List of labels corresponding to plots.
    figsize : tuple
        Figsize for two axes plots.
    """
    def update(wellname, rate, cumulative, time_start, time_end):
        rates = wells[wellname].total_rates

        _, ax = plt.subplots(1, 1, figsize=figsize)
        title = 'Cumulative ' + rate if cumulative else rate
        ax.set_title('Well {}. {}'.format(wellname.upper(), title))
        ax.set_ylabel('Cumulative Rate' if cumulative else 'Rate')
        ax.set_xlabel('Date')
        t = rates.loc[time_start:time_end, 'DATE']
        data = _safe_data_getter(rates, time_start, time_end, NAMES_TO_KW.get(rate, rate))
        if cumulative:
            data = np.cumsum(data)
        ax.plot(t, data, 'o-', c=COLORS[rate], label=labels[0], ms=3.5, lw=1.3)
        ax.axvline(rates.loc[time_start, 'DATE'], ls=':', c='grey')
        ax.axvline(rates.loc[time_end, 'DATE'], ls=':', c='grey')
        ax.grid(True)

        if wells2 is not None:
            rates2 = wells2[wellname].total_rates
            t = rates2.loc[time_start:time_end, 'DATE']
            data = _safe_data_getter(rates2, time_start, time_end, NAMES_TO_KW.get(rate, rate))
            if cumulative:
                data = np.cumsum(data)
            ax.plot(t, data, 'o-', c='green', label=labels[1], ms=3.5, lw=1.3)
            ax.legend(loc='best')

    timesteps_len = len(timesteps) - 1

    if labels is None:
        labels = ['Model_1', 'Model_2'] if wells2 is not None else ['']

    interact(update,
             wellname=widgets.Dropdown(options=wellnames),
             rate=widgets.Dropdown(options=['Oil', 'Water', 'Gas', 'Free Gas'], value='Oil'),
             cumulative=widgets.Checkbox(value=False, description='Cumulative'),
             time_start=widgets.IntSlider(min=0, max=timesteps_len, step=1, value=0),
             time_end=widgets.IntSlider(min=0, max=timesteps_len, step=1, value=timesteps_len))
    plt.show()

def show_rates2(wells, timesteps, wellnames, wells2=None, labels=None, figsize=(16, 6)):
    """Plot total liquid and gas rates for a chosen well segment on two separate axes.

    Parameters
    ----------
    wells : Wells class instance
        Wells component with calculated rates in results attribute.
    timesteps : list of Timestamps
        Dates at which rates were calculated.
    wellnames : array-like
        List of active producing wells.
    wells2 : Wells class instance
        Wells component with results for comparison.
    labels : array-like
        List of labels corresponding to plots.
    figsize : tuple
        Figsize for two axes plots.
    """
    def update(wellname, liquid_rate, gas_rate, cumulative, time_start, time_end):
        rates = wells[wellname].total_rates

        _, axes = plt.subplots(1, 2, figsize=figsize)
        title = 'Cumulative ' + liquid_rate if cumulative else liquid_rate
        axes[0].set_title('Well {}. {}'.format(wellname.upper(), title))
        axes[0].set_ylabel('Cumulative Rate' if cumulative else 'Rate')
        axes[0].set_xlabel('Date')
        t = rates.loc[time_start:time_end, 'DATE']
        data = _safe_data_getter(rates, time_start, time_end, NAMES_TO_KW.get(liquid_rate, liquid_rate))
        if cumulative:
            data = np.cumsum(data)
        axes[0].plot(t, data, 'o-', c=COLORS[liquid_rate], label=labels[0], ms=3.5, lw=1.3)
        axes[0].axvline(rates.loc[time_start, 'DATE'], ls=':', c='grey')
        axes[0].axvline(rates.loc[time_end, 'DATE'], ls=':', c='grey')
        axes[0].grid(True)

        title = 'Cumulative ' + gas_rate if cumulative else gas_rate
        axes[1].set_title('Well {}. {}'.format(wellname.upper(), title))
        axes[1].set_ylabel('Cumulative Rate' if cumulative else 'Rate')
        data = _safe_data_getter(rates, time_start, time_end, NAMES_TO_KW.get(gas_rate, gas_rate))
        if cumulative:
            data = np.cumsum(data)
        axes[1].plot(t, data, 'o-', c=COLORS[gas_rate], label=labels[0], ms=3.5, lw=1.3)
        axes[1].axvline(rates.loc[time_start, 'DATE'], ls=':', c='grey')
        axes[1].axvline(rates.loc[time_end, 'DATE'], ls=':', c='grey')
        axes[1].grid(True)

        if wells2 is not None:
            rates2 = wells2[wellname].total_rates
            t = rates2.loc[time_start:time_end, 'DATE']
            data = _safe_data_getter(rates2, time_start, time_end, NAMES_TO_KW.get(liquid_rate, liquid_rate))
            if cumulative:
                data = np.cumsum(data)
            axes[0].plot(t, data, 'o-', c=COMPARE_COLOR, label=labels[1], ms=3.5, lw=1.3)
            data = _safe_data_getter(rates2, time_start, time_end, NAMES_TO_KW.get(gas_rate, gas_rate))
            if cumulative:
                data = np.cumsum(data)
            axes[1].plot(t, data, 'o-', c=COMPARE_COLOR, label=labels[1], ms=3.5, lw=1.3)
            axes[0].legend(loc='best')
            axes[1].legend(loc='best')

    timesteps_len = len(timesteps) - 1
    if labels is None:
        labels = ['Model_1', 'Model_2'] if wells2 is not None else ['']

    interact(update,
             wellname=widgets.Dropdown(options=wellnames),
             liquid_rate=widgets.Dropdown(options=['Oil', 'Water'], value='Oil'),
             gas_rate=widgets.Dropdown(options=['Gas', 'Free Gas'], value='Gas'),
             cumulative=widgets.Checkbox(value=False, description='Cumulative'),
             time_start=widgets.IntSlider(min=0, max=timesteps_len, step=1, value=0),
             time_end=widgets.IntSlider(min=0, max=timesteps_len, step=1, value=timesteps_len))
    plt.show()
