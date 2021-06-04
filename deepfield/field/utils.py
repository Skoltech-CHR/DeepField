"""Miscellaneous utils."""
import os
import re
import subprocess
import signal
import fnmatch
from contextlib import contextmanager
from anytree import PreOrderIter
import numpy as np
import pandas as pd
import vtk
import psutil
from tqdm import tqdm

try:
    from credentials import TNAV_LICENSE_URL
except ImportError:
    TNAV_LICENSE_URL = None


@contextmanager
def _dummy_with():
    """Dummy statement."""
    yield

def kill(proc_pid):
    """Kill proc and its childs."""
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

def signal_handler(signum, frame):
    """Timeout handler."""
    _ = signum, frame
    raise Exception("Timed out!")

def execute_tnav_models(base_script_path, models, license_url=TNAV_LICENSE_URL, logfile=None,
                        global_timeout=None, process_timeout=None):
    """Execute a bash script for each model in a set of models.

    Parameters
    ----------
    base_script_path : str
        Path to script to execute.
    models : str, list of str
        A path to model or list of pathes.
    license_url : str
        A license server url.
    logfile : str
        A path to file where to point stdout and stderr.
    global_timeout : int
        Global timeout in seconds.
    process_timeout : int
        Process timeout. Kill process that exceeds the timeout and go to the next model.
    """
    if license_url is None:
        raise ValueError('License url is not defined.')
    models = np.atleast_1d(models)
    base_args = ['bash', base_script_path, license_url]
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(-1 if global_timeout is None else global_timeout)
    with (open(logfile, 'w') if logfile is not None else _dummy_with()) as f:
        for model in tqdm(models):
            try:
                p = subprocess.Popen(base_args + [model], stdout=f, stderr=f)#pylint:disable=consider-using-with
                try:
                    p.wait(timeout=process_timeout)
                except subprocess.TimeoutExpired:
                    kill(p.pid)
            except Exception as err:
                kill(p.pid)
                raise err

def recursive_insensitive_glob(path, pattern, return_relative=False):
    """Find files matching pattern ignoring case-style."""
    found = []
    reg_expr = re.compile(fnmatch.translate(pattern), re.IGNORECASE)
    for root, _, files in os.walk(path, topdown=True):
        for f in files:
            if re.match(reg_expr, f):
                f_path = os.path.join(root, f)
                found.append(
                    f_path if not return_relative else os.path.relpath(f_path, start=path)
                )
    return found

def get_single_path(dir_path, filename, logger=None):
    """Find a file withihn the directory. Raise error if multiple files found."""
    files = recursive_insensitive_glob(dir_path, filename)
    if not files:
        if logger is not None:
            logger.warning("{} file was not found.".format(filename))
        return None
    if len(files) > 1:
        raise ValueError('Directory {} contains multiple {} files.'.format(dir_path, filename))
    return files[0]

def hasnested(container, *chain):
    """Checks if chain contains in container.

    Parameters
    ----------
    container: `collections.abc.Container`
    chain: tuple
        List of keywords.

    Returns
    -------
    out: bool
        True if `chain[0]` in container and `chain[1]` in `container[chain[0]]` etc.
        or if chain is empty, else False.
    """
    key, chain = chain[0], chain[1:]
    if key in container:
        return hasnested(container[key], *chain) if chain else True
    return False

def rolling_window(a, strides):
    """Rolling window without overlays."""
    strides = np.asarray(strides)
    output_shape = tuple((np.array(a.shape) - strides)//strides + 1) + tuple(strides)
    output_strides = tuple(strides * np.asarray(a.strides)) + a.strides
    return np.lib.stride_tricks.as_strided(a, shape=output_shape, strides=output_strides)

def mk_vtk_id_list(id_list):
    """Complementary function."""
    vil = vtk.vtkIdList()
    for i in id_list:
        vil.InsertNextId(int(i))
    return vil

def length_segment(p1, p2):
    """Calculates dist between two 3d points."""
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    return np.sqrt(np.sum((p1-p2)**2))

def get_point_on_line_at_distance(p1, p2, distance):
    """ Finds point on line going through p1 and p2 on given distance from p1."""
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    distance_ratio = distance / length_segment(p1, p2)
    return p1 + distance_ratio*(p2-p1)

def get_well_mask(field):
    """Get the model's well mask in a spatial form.

    Parameters
    ----------
    field: Field

    Returns
    -------
    well_mask: np.array
        Array with well-names in cells which are registered as well-blocks and empty strings everywhere else.
    """
    if field.state.spatial:
        well_mask = np.zeros(field.grid.dimens, dtype='U32')
    else:
        well_mask = np.zeros(field.grid.actnum.sum(), dtype='U32')
    for node in field.wells:
        if node.is_main_branch:
            for branch in PreOrderIter(node):
                ind = branch.blocks
                if ind.shape[0]:
                    if field.wells.state.spatial:
                        ind = ind.T
                        well_mask[ind[0], ind[1], ind[2]] = branch.name
                    else:
                        well_mask[ind] = branch.name
    return well_mask

def full_ind_to_active_ind(ind, grid):
    """Transforms 1D indices with respect to all cells to the indices with respect to only active cells.

    Parameters
    ----------
    ind: array-like
        Indices to be transformed.
    grid: Grid
        Grid component of a model.

    Returns
    -------
    ind: array-like
        Transformed indices.
    """
    ind = ind.copy()
    f2a = grid.ravel(attr='actnum', inplace=False).astype(np.int)
    ind[f2a[ind] == 0] = -1
    f2a[f2a == 1] = np.arange(f2a.sum())
    f2a = np.concatenate([f2a, [-1]])
    return f2a[ind]

def active_ind_to_full_ind(ind, grid):
    """Transforms 1D indices with respect to active to the indices with respect to all cells.

    Parameters
    ----------
    ind: array-like
        Indices to be transformed.
    grid: Grid
        Grid component of a model.

    Returns
    -------
    ind: array-like
        Transformed indices.
    """
    actnum = grid.ravel(attr='actnum', inplace=False).astype(np.int)
    a2f = np.arange(len(actnum))[actnum == 1]
    a2f = np.concatenate([a2f, [-1]])
    return a2f[ind]

def get_control_interval_mask(control_dates, time_interval):
    """Returns a mask for control dates that affect the given time interval."""
    mask_prehistory = control_dates <= time_interval[0]
    if not any(mask_prehistory):
        raise ValueError('First control date {} is later than the beginning of time interval ({}, {}).'
                         .format(control_dates[0], time_interval[0], time_interval[1]))
    first_control_date = control_dates[mask_prehistory][-1]
    mask = (control_dates >= first_control_date) & (control_dates < time_interval[1])
    return mask


def get_control_interval_dates(field, time_interval=None):
    """Get the dates of the control changes in the given interval."""
    dates = field.wells.event_dates
    prehistory_dates = []
    if time_interval is not None:
        mask_prehistory = dates <= time_interval[0]
        if any(mask_prehistory):
            first_control_date = dates[mask_prehistory][-1]
        else:
            first_control_date = dates[0]
            prehistory_dates.append(time_interval[0])
        mask = (dates >= first_control_date) & (dates < time_interval[1])
        dates = dates[mask]
    return pd.to_datetime(prehistory_dates), dates

def get_spatial_well_control(field, attrs, date_range=None, fill_shut=0., fill_outside=0.):
    """Get the model's control in a spatial. Also returns control dates relative to model start date.

    Parameters
    ----------
    field: Field
        Geological model.
    attrs: tuple or list
        Control attributes to get data from.
    date_range: tuple
        Minimal and maximal dates for control events.
    fill_shut: float
        Value to fill closed perforations
    fill_outside:
        Value to fill non-perforated cells

    Returns
    -------
    control: np.array
    """
    spatial = field.state.spatial
    well_mask = field.well_mask
    attrs = [k.upper() for k in attrs]

    prehistory_dates, dates = get_control_interval_dates(field, date_range)

    spatial_dims = tuple(field.grid.dimens) if spatial else (np.sum(field.grid.actnum),)

    control = np.full((len(prehistory_dates) + len(dates), len(attrs)) + spatial_dims, fill_outside)

    for node in field.wells:
        if node.is_main_branch and 'EVENTS' in node:
            df = pd.DataFrame(fill_shut, index=dates, columns=attrs)
            df.loc[node.events['DATE'], attrs] = node.events[attrs].values
            df = df.fillna(fill_shut)
            if fill_shut:
                df = df.replace(0, fill_shut)
            for branch in PreOrderIter(node):
                control[len(prehistory_dates):, ..., well_mask == branch.name] = np.expand_dims(df.values, -1)
                control[:len(prehistory_dates), ..., well_mask == branch.name] = fill_shut

    sec_in_day = 86400
    dates = prehistory_dates.union(dates)
    rel_dates = (pd.to_datetime(dates) - field.start).total_seconds().values / sec_in_day
    return {'control': control, 't': rel_dates}

def _remove_repeating_blocks(blocks, values=None):
    if not len(blocks):
        if values is not None:
            return blocks, values
        return blocks
    new_blocks = []
    if values is not None:
        new_values = []
    for i, p in enumerate(blocks):
        if blocks.ndim == 1:
            occurrences = np.where((p == blocks))[0]
        else:
            occurrences = np.where((p == blocks).all(axis=1))[0]
        if len(occurrences) > 1:
            if i != occurrences[0]:
                continue
            if values is not None:
                for ind in occurrences[1:]:
                    i = ind if values[ind] > values[i] else i
        new_blocks.append(p)
        if values is not None:
            new_values.append(values[i])
    new_blocks = np.stack(new_blocks)
    new_values = np.stack(new_values)
    return new_blocks, new_values


# pylint: disable=too-many-nested-blocks
def get_spatial_perf(field, subset=None, mode=None):
    """Get model's perforation ratios in a spatial form.

    Parameters
    ----------
    field: Field
    subset: array-like or None
        Subset of timesteps to pick. If None, picks all timesteps available.
    mode: str, None
        If not None, pick the blocks only with specified mode.

    Returns
    -------
    perf_ratio: np.array
    """
    spatial = field.state.spatial
    full_perforation = field.wells.state.full_perforation
    if subset is None:
        n_ts = len(field.wells.event_dates)
    else:
        n_ts = len(subset) - 1
    spatial_dims = tuple(field.grid.dimens) if spatial else (np.sum(field.grid.actnum),)
    perf = np.zeros((n_ts, 1) + spatial_dims)
    event_dates = field.wells.event_dates
    for t in range(n_ts):
        field.wells.apply_perforations(event_dates[t])
        for well in field.wells:
            if well.is_main_branch:
                if mode is not None:
                    if not hasattr(well, 'events'):
                        continue
                    mode_at_ts = well.events[well.events['DATE'] == event_dates[t]]['MODE']
                    if len(mode_at_ts) == 0 or not (mode_at_ts == mode.upper()).all():
                        continue
                for branch in PreOrderIter(well):
                    perf_ind_mask = branch.perforated_indices()
                    perf_ind = branch.blocks[perf_ind_mask]
                    perf_ratio = branch.blocks_info.PERF_RATIO[perf_ind_mask].values
                    perf_ind, perf_ratio = _remove_repeating_blocks(perf_ind, perf_ratio)
                    if perf_ind.shape[0]:
                        if spatial:
                            perf[t, 0, perf_ind[:, 0], perf_ind[:, 1], perf_ind[:, 2]] = perf_ratio
                        else:
                            perf[t, 0, perf_ind] = perf_ratio
    if full_perforation:
        field.wells.apply_perforations()
    return perf

# pylint: disable=too-many-nested-blocks
def get_spatial_cf_and_perf(field, subset=None, mode=None):
    """Get model's connection factors and perforation ratios in a spatial form.

    Parameters
    ----------
    field: Field
    subset: array-like or None
        Subset of timesteps to pick. If None, picks all timesteps available.
    mode: str, None
        If not None, pick the blocks only with specified mode.

    Returns
    -------
    connection_factors: np.array
    perf_ratio: np.array
    """
    spatial = field.state.spatial
    full_perforation = field.wells.state.full_perforation
    if subset is None:
        n_ts = len(field.wells.event_dates)
    else:
        n_ts = len(subset) - 1

    spatial_dims = tuple(field.grid.dimens) if spatial else (np.sum(field.grid.actnum),)

    cf = np.zeros((n_ts, 1) + spatial_dims)
    perf = np.zeros((n_ts, 1) + spatial_dims)

    event_dates = field.wells.event_dates

    for t in range(n_ts):
        field.wells.apply_perforations(event_dates[t])
        field.wells.calculate_cf(field.rock, field.grid, units=field.meta.get('UNITS', 'METRIC'))
        for well in field.wells:
            if well.is_main_branch:
                if mode is not None:
                    if not hasattr(well, 'events'):
                        continue
                    mode_at_ts = well.events[well.events['DATE'] == event_dates[t]]['MODE']
                    if len(mode_at_ts) == 0 or not (mode_at_ts == mode.upper()).all():
                        continue
                for branch in PreOrderIter(well):
                    perf_ind_mask = branch.perforated_indices()
                    perf_ind = branch.blocks[perf_ind_mask]
                    perf_ratio = branch.blocks_info.PERF_RATIO[perf_ind_mask].values
                    perf_ind, perf_ratio = _remove_repeating_blocks(perf_ind, perf_ratio)

                    cf_ind = branch.blocks
                    connection_factors = branch.blocks_info.CF.values
                    cf_ind, connection_factors = _remove_repeating_blocks(cf_ind, connection_factors)

                    if perf_ind.shape[0]:
                        if spatial:
                            perf[t, 0, perf_ind[:, 0], perf_ind[:, 1], perf_ind[:, 2]] = perf_ratio
                            cf[t, 0, cf_ind[:, 0], cf_ind[:, 1], cf_ind[:, 2]] = connection_factors
                        else:
                            perf[t, perf_ind] = perf_ratio
                            cf[t, perf_ind] = connection_factors
    if full_perforation:
        field.wells.apply_perforations()
    return cf, perf


def get_n_control_ts(model):
    """Get number of timesteps in the model's control variable"""
    return len(model.wells.event_dates)


def overflow_safe_mean(arr, axis=None):
    """Computes mean values across an array with reduced overflow risk (NO WARRANTIES, THOUGH - STILL MAY OVERFLOW).

    Parameters
    ----------
    arr: array-like
    axis: int or tuple, optional

    Returns
    -------
    mean: array-like
    """
    if axis is None:
        return arr.mean()
    if isinstance(axis, int):
        return arr.mean(axis=axis)
    for ax in sorted(axis, reverse=True):
        arr = arr.mean(axis=ax)
    return arr
