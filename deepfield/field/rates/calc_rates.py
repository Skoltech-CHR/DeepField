"""Methods for rates calculation"""
import multiprocessing
from multiprocessing.connection import wait
import numpy as np
import pandas as pd
from IPython.display import clear_output
from ..tables.table_interpolation import baker_linear_model

def find_control(model, curr_date, wname):
    """Find well control from events.

    Parameters
    ----------
    model : Field class instance
        Reservoir model.
    curr_date : pandas.Timestamp
        Date to find control for.
    wname : str
        Well segment name.

    Returns
    -------
    p_bh : float
        Bottomhole pressure of well segment (0 means well is shut in or INJ).
    well_mode : str
        Well segment mode (PROD - production, INJ - injection, '' - undefined).
    """
    well = model.wells[wname.split(':')[0]]
    p_bh = 0
    well_mode = ''
    status = 'CLOSED'
    if (not 'EVENTS' in well) or well.events.empty:
        return p_bh, well_mode, status
    mask = well.events['DATE'] < curr_date
    if not mask.any():
        return p_bh, well_mode, status
    last = np.where(mask)[0].max()
    well_mode = well.events.loc[last, 'MODE']
    if 'PROD' in well_mode:
        p_bh = well.events.loc[last, 'BHPT']
        status = 'OPEN'
    return p_bh, well_mode, status

def rates_oil_disgas(model, time_ind, curr_date, wellname,
                     units, g_const, cf_aggregation='sum'):
    """Calculate oil/water/gas rates and pvt properties for given timestep.

    Returns
    -------
    rates : pd.DataFrame
        pd.Dataframe filled with production rates for each phase.
    block_rates : pd.DataFrame of ndarrays
        pd.Dataframe filled with arrays of production rates in each grid block.
    """
    _ = g_const
    well = model.wells[wellname]
    results = model.wells[wellname].results
    dynamics = model.wells[wellname].blocks_dynamics

    well.apply_perforations(current_date=curr_date)
    well.calculate_cf(model.rock, model.grid, units=units, cf_aggregation=cf_aggregation)

    conn_factors = well.blocks_info['CF'][well.blocks_info['PERF_RATIO'] > 0.].values
    p_bh, well_mode, status = find_control(model, curr_date, wellname)
    perf_blocks = well.perforated_blocks()
    perf_blocks_indices = well.blocks_info[well.blocks_info['PERF_RATIO'] > 0].index
    results.loc[time_ind, ['MODE', 'STATUS']] = well_mode, status
    dynamics.loc[time_ind, ['MODE', 'STATUS']] = well_mode, status
    if ((perf_blocks.shape[0] == 0) or ('PROD' not in well_mode)
            or (time_ind == 0) or (status == 'CLOSED')):
        return
    x_blocks, y_blocks, z_blocks = perf_blocks.T
    pressure = model.states.pressure[time_ind, x_blocks, y_blocks, z_blocks]
    soil = model.states.soil[time_ind, x_blocks, y_blocks, z_blocks]
    swat = model.states.swat[time_ind, x_blocks, y_blocks, z_blocks]
    sgas = 1 - soil - swat
    rs = model.states.rs[time_ind, x_blocks, y_blocks, z_blocks]

    fvf_o, mu_o = model.tables.pvto(np.vstack((rs, pressure)).T).T
    fvf_w, mu_w = model.tables.pvtw(pressure).T
    fvf_g, mu_g = model.tables.pvdg(pressure).T

    kr_w = model.tables.swof(swat)[:, 0]
    kr_g = model.tables.sgof(sgas)[:, 0]
    swc = model.tables.swof.index.values[0]
    kr_o = baker_linear_model(model.tables, swat, sgas, swc)

    oil_rate = (conn_factors*kr_o/mu_o*(pressure - p_bh))/fvf_o
    oil_rate[oil_rate < 0.] = 0.
    water_rate = (conn_factors*kr_w/mu_w*(pressure - p_bh))/fvf_w
    water_rate[water_rate < 0.] = 0.
    free_gas = (conn_factors*kr_g/mu_g*(pressure - p_bh))/fvf_g
    free_gas[free_gas < 0.] = 0.
    gas_rate = free_gas + oil_rate*rs
    for col, data in zip(dynamics.columns[3:],
                         [p_bh, oil_rate, water_rate, gas_rate, free_gas,
                          fvf_o, fvf_w, fvf_g, mu_o, mu_w, mu_g, kr_o, kr_w, kr_g]):
        data2 = np.zeros(len(well.blocks))
        data2[perf_blocks_indices] = data
        dynamics.at[time_ind, col] = data2
    results.loc[time_ind, 'WBHP'] = p_bh
    results.loc[time_ind, ['WOPR', 'WWPR']] = np.sum(oil_rate), np.sum(water_rate)
    results.loc[time_ind, ['WGPR', 'WFGPR']] = np.sum(gas_rate), np.sum(free_gas)
    return

def launch_calculus(model, timesteps, wellname, cf_aggregation='sum', queue=None):
    """Calculate production rates.

    Returns
    -------
    rates : pd.DataFrame
        pd.Dataframe filled with production rates for each phase.
    block_rates : pd.DataFrame of ndarrays
        pd.Dataframe filled with arrays of production rates in each grid block.
    cf_aggregation: str, 'sum' or 'eucl'
        The way of aggregating cf projection ('sum' - sum, 'eucl' - Euclid norm).
    """
    empty_results = pd.DataFrame(columns=['DATE', 'MODE', 'STATUS', 'WBHP',
                                          'WOPR', 'WWPR', 'WGPR', 'WFGPR'])
    empty_dynamics = pd.DataFrame(columns=['DATE', 'MODE', 'STATUS', 'WBHP',
                                           'WOPR', 'WWPR', 'WGPR', 'WFGPR',
                                           'FVF_O', 'FVF_W', 'FVF_G',
                                           'VISC_O', 'VISC_W', 'VISC_G',
                                           'KR_O', 'KR_W', 'KR_G'])
    units = model.meta.get('UNITS', 'METRIC')
    g_const = 0.0000980665 if units == 'METRIC' else 0.00694
    fluids = set(model.meta['FLUIDS'])

    well = model.wells[wellname]
    setattr(well, 'RESULTS', empty_results)
    setattr(well, 'BLOCKS_DYNAMICS', empty_dynamics)
    well.results['DATE'] = timesteps
    well.blocks_dynamics['DATE'] = timesteps
    well.results.loc[:, empty_results.columns[3:]] = 0.
    for col in empty_dynamics.columns[3:]:
        for t_ind in range(len(timesteps)):
            well.blocks_dynamics.at[t_ind, col] = np.zeros(len(well.blocks))
    if len(well.blocks_info) == 0:
        return
    if 'PERF' in well.attributes:
        if len(well.perf) == 0:
            return
    well.blocks_info['PERF_RATIO'] = 0.
    if fluids == set(('OIL', 'WATER', 'GAS', 'DISGAS')):
        for t, curr_date in enumerate(timesteps):
            rates_oil_disgas(model, t, curr_date, wellname, units, g_const, cf_aggregation)

    if queue is not None:
        queue.send(((wellname, well.results, well.blocks_dynamics),
                    multiprocessing.current_process().name))
        queue.close()

#pylint: disable=too-many-branches
def calc_rates_multiprocess(model, timesteps, wellnames, cf_aggregation='sum', verbose=True):
    """Run multiprocessed calculation of rates."""
    readers = []
    for name in wellnames:
        r, w = multiprocessing.Pipe(duplex=False)
        readers.append(r)
        p = multiprocessing.Process(target=launch_calculus,
                                    args=(model, timesteps, name, cf_aggregation, w))
        p.start()
        w.close()
    reader_counter = 0
    while readers:
        for r in wait(readers):
            try:
                msg = r.recv()
                setattr(model.wells[msg[0][0]], 'RESULTS', msg[0][1])
                setattr(model.wells[msg[0][0]], 'BLOCKS_DYNAMICS', msg[0][2])
            except EOFError:
                readers.remove(r)
            else:
                reader_counter += 1
                if verbose:
                    clear_output(wait=True)
                    print(f'Processed {reader_counter} out of {len(wellnames)} wells')
    return model

def calc_rates(model, timesteps, wellnames, cf_aggregation='sum', verbose=True):
    """Calculate production rates.

    Returns
    -------
    rates : pd.DataFrame
        pd.Dataframe filled with production rates for each phase.
    block_rates : pd.DataFrame of ndarrays
        pd.Dataframe filled with arrays of production rates in each grid block.
    cf_aggregation: str, 'sum' or 'eucl'
        The way of aggregating cf projection ('sum' - sum, 'eucl' - Euclid norm).
    """
    for i, wellname in enumerate(wellnames):
        launch_calculus(model, timesteps, wellname, cf_aggregation)
        if verbose:
            clear_output(wait=True)
            print(f'Processed {i+1} out of {len(wellnames)} wells')
    return model
