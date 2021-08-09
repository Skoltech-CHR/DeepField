"""SUMMARY assisting functions."""

import os
import numpy as np

from .share import CHAR, BlockField, INTE, SpecElement, write_unrst_section, write_unrst_data_section, \
    REAL, format_keyword

from ..parse_utils.ecl_binary import _read_sections

RESTART = 'RESTART'
DIMENS = 'DIMENS'
KEYWORDS = 'KEYWORDS'
WGNAMES = 'WGNAMES'
NUMS = 'NUMS'
UNITS = 'UNITS'
STARTDAT = 'STARTDAT'
SEQHDR = 'SEQHDR'
MINISTEP = 'MINISTEP'
PARAMS = 'PARAMS'

FIELD = 'FIELD'
TIME = 'TIME'
DAY = 'DAY'
DAYS = 'DAYS'
MONTH = 'MONTH'
MONTHS = 'MONTHS'
YEAR = 'YEAR'
YEARS = 'YEARS'
WWPR = 'WWPR'
WOPR = 'WOPR'
WGPR = 'WGPR'
NONE = ':+:+:+:+'


SUMMARY_EXT = '.S{time:0>4}'
UNIFIED_SUMMARY_EXT = '.UNSMRY'
INDEX_SUMMARY_EXT = '.SMSPEC'

STARTDAT_SIZE = 6


# Header's block formats
INDEX_META_BLOCK_SPEC = {
    RESTART: {
        'type': CHAR,
        'items_cnt': 9,
        'struct': {
            'item1': BlockField(idx=1, default=b' ' * 8),
            'item2': BlockField(idx=2, default=b' ' * 8),
            'item3': BlockField(idx=3, default=b' ' * 8),
            'item4': BlockField(idx=4, default=b' ' * 8),
            'item5': BlockField(idx=5, default=b' ' * 8),
            'item6': BlockField(idx=6, default=b' ' * 8),
            'item7': BlockField(idx=7, default=b' ' * 8),
            'item8': BlockField(idx=8, default=b' ' * 8),
            'item9': BlockField(idx=9, default=b' ' * 8)
        }
    },
    DIMENS: {
        'type': INTE,
        'items_cnt': 6,
        'struct': {
            'nlist': BlockField(idx=1, default=-1),
            'dimensions': BlockField(idx=2, cnt=3),
            'dummy1': BlockField(idx=5, default=-1),
            'dummy2': BlockField(idx=6, default=-1)
        }
    }
}

INDEX_SECTIONS_DATA = {
    KEYWORDS: SpecElement(KEYWORDS, CHAR),
    WGNAMES: SpecElement(WGNAMES, CHAR),
    NUMS: SpecElement(NUMS, INTE),
    UNITS: SpecElement(UNITS, CHAR),
    STARTDAT: SpecElement(STARTDAT, INTE)
}

DATA_BLOCK_SPEC = {
    SEQHDR: SpecElement(SEQHDR, INTE, 1),
    MINISTEP: SpecElement(MINISTEP, INTE, 1),
    PARAMS: SpecElement(PARAMS, REAL)
}

UNITS_DATA = {
    TIME: DAYS,
    YEARS: YEARS,
    DAY: DAYS,
    MONTH: MONTHS,
    YEAR: YEARS,
    WWPR: 'SM3/DAY',
    WOPR: 'SM3/DAY',
    WGPR: 'SM3/DAY'
}

def get_summary_file(is_unified, name, time, unified_file, mode):
    """
    Get file object for summary file

    Parameters
    ----------
    is_unified : bool
        If true, the target file will be in unified format.
        Otherwise the target file will be in non unified format.
    time:
        Time

    Returns
    -------
    out : file object

    Notes
    -----
    If is_unified is True the function return the same file.
    If is_unified is False the function return the new file each time.

    """
    if is_unified:
        if unified_file:
            return unified_file, unified_file
        unified_file = open(name + UNIFIED_SUMMARY_EXT, f'{mode}+b') #pylint: disable=consider-using-with
        return unified_file, unified_file
    return open(name + SUMMARY_EXT.format(time=time), f'{mode}+b'), unified_file#pylint: disable=consider-using-with


def get_startdat_section_data(date):
    """
    Get data for STARTDAT section

    Returns
    -------
    out : np.array
        Array of items

    See Also
    --------
    np.array

    Examples
    --------
    >>> l = get_startdat_section_data()

    """
    return np.array([date.day, date.month, date.year, 0, 0, 0])


def get_keywords_section_data(rates):
    """
    Get data for STARTDAT section

    Parameters
    ----------

    Returns
    -------
    out : np.array
        Array of items

    See Also
    --------
    np.array

    """
    keywords_list = [format_keyword(TIME), format_keyword(YEARS), format_keyword(DAY), format_keyword(MONTH),
                     format_keyword(YEAR)]
    nums_list = [0, 0, 0, 0, 0]
    wgname_keys = list(rates.keys())
    wgname_keys2 = []
    wgname_keys2.extend(wgname_keys)
    if FIELD in wgname_keys2:
        wgname_keys2.remove(FIELD)
    for key in wgname_keys:
        names = list(map(lambda item: format_keyword(item), rates[key].keys()))
        keywords_list.extend(names)
        if key == FIELD:
            nums_list.extend([0] * len(names))
        else:
            nums_list.extend([wgname_keys2.index(key) + 1] * len(names))
    nlist = len(keywords_list)
    return np.array(keywords_list), np.array(nums_list), nlist


def get_wgnames_section_data(rates):
    """
    Get data for WGNAMES section

    Parameters
    ----------

    Returns
    -------
    out : np.array
        Array of items

    See Also
    --------
    np.array

    """
    wgname_list = [NONE, NONE, NONE, NONE, NONE]
    wgname_keys = list(rates.keys())
    for key in wgname_keys:
        names = list(map(lambda item: format_keyword(item), rates[key].keys()))
        wgname_list.extend([key] * len(names))
    return np.array(wgname_list)


def get_units_section_data(rates):
    """
    Get data for UNITS section

    Parameters
    ----------

    Returns
    -------
    out : np.array
        Array of items

    See Also
    --------
    np.array

    """
    keywords_list, _, nlist = get_keywords_section_data(rates)
    return np.array(list(map(lambda key: format_keyword(UNITS_DATA[key.strip()]), keywords_list))), nlist


def get_params_section_data(rates, date, time_idx, time):
    """
    Get data for PARAMS section

    Parameters
    ----------
    time : int
        Time interval

    Returns
    -------
    out : np.array
        Array of items

    See Also
    --------
    np.array

    """
    result_data = [float(time), float(time) / 365., date.day, date.month, date.year]
    wgname_keys = list(rates.keys())

    for wgkey in wgname_keys:
        for mnemonic in rates[wgkey].keys():
            result_data.append(rates[wgkey][mnemonic][time_idx])

    return np.array(result_data)


def get_time_size(rates):
    """
    Get number of time intervals

    Parameters
    ----------

    Returns
    -------
    out : int
        Number of time intervals

    See Also
    --------
    DataStruct

    """
    wgname_keys = list(rates.keys())
    mnemo_keys = list(rates[wgname_keys[0]].keys())
    return len(rates[wgname_keys[0]][mnemo_keys[0]])


def save_index_summary(name, rates, dates, grid_dim):
    """
    Save index file

    Parameters
    ----------


    See Also
    --------
    DataStruct

    """
    with open(name + INDEX_SUMMARY_EXT, "w+b") as file_index:
        nlist = 0
        keywords_data, nums_data, nlist = get_keywords_section_data(rates)  # need to calc NLIST filed for DIMENS
        write_unrst_data_section(f=file_index, name=RESTART, stype=INDEX_META_BLOCK_SPEC[RESTART]['type'],
                                 data_array=np.array(
                                     [' ' * 8, ' ' * 8, ' ' * 8, ' ' * 8, ' ' * 8, ' ' * 8, ' ' * 8, ' ' * 8, ' ' * 8]))
        dimen = INDEX_META_BLOCK_SPEC[DIMENS]
        dimen['struct']['nlist'].val = nlist
        write_unrst_section(file_index, DIMENS, dimen, grid_dim)
        write_unrst_data_section(f=file_index, name=KEYWORDS, stype=INDEX_SECTIONS_DATA[KEYWORDS].type,
                                 data_array=keywords_data)
        wgnames_date = get_wgnames_section_data(rates)
        write_unrst_data_section(f=file_index, name=WGNAMES, stype=INDEX_SECTIONS_DATA[WGNAMES].type,
                                 data_array=wgnames_date)
        write_unrst_data_section(f=file_index, name=NUMS, stype=INDEX_SECTIONS_DATA[NUMS].type,
                                 data_array=nums_data)
        units_data, nlist = get_units_section_data(rates)
        write_unrst_data_section(f=file_index, name=UNITS, stype=INDEX_SECTIONS_DATA[UNITS].type,
                                 data_array=units_data)
        write_unrst_data_section(f=file_index, name=STARTDAT, stype=INDEX_SECTIONS_DATA[STARTDAT].type,
                                 data_array=get_startdat_section_data(dates[0]))

    return nlist

def save_summary_bytime(f, rates, date, time_idx, time):
    """
    Save summary section by time segment

    Parameters
    ----------
    f : file object
    time : int
        Time interval

    See Also
    --------
    DataStruct

    """
    write_unrst_data_section(f=f, name=SEQHDR, stype=DATA_BLOCK_SPEC[SEQHDR].type,
                             data_array=np.array([0]))
    write_unrst_data_section(f=f, name=MINISTEP, stype=DATA_BLOCK_SPEC[MINISTEP].type,
                             data_array=np.array([0]))
    params_data = get_params_section_data(rates, date, time_idx, time)
    write_unrst_data_section(f=f, name=PARAMS, stype=DATA_BLOCK_SPEC[PARAMS].type,
                             data_array=params_data)


def save_summary_file(is_unified, rates, dates, name, mode, start_idx, start_t):
    """
    Save summary file

    Parameters
    ----------
    is_unifie

    See Also
    --------
    DataStruct

    """

    file_summary = None
    time_size = get_time_size(rates)

    unified_file = None
    for time in range(start_idx, time_size):
        file_summary, unified_file = get_summary_file(is_unified, name, time, unified_file, mode)
        save_summary_bytime(file_summary, rates, dates[time].date(), time, start_t + time)
        if not is_unified:
            file_summary.close()
    if file_summary and not file_summary.closed:
        file_summary.close()

def save_summary(is_unified, name, rates, dates, grid_dim, mode, logger=None):
    """
    Save summary and index file

    Parameters
    ----------
    is_unified

    See Also
    --------
    DataStruct

    """

    def logger_print(msg, level='warning'):
        if logger is not None:
            getattr(logger, level)(msg)

    start_idx = 1
    start_t = 0
    if (mode == 'w' or\
        (mode == 'a' and\
         (not os.path.isfile(name + '.UNSMRY') or not os.path.isfile(name + '.SMSPEC')))):
        mode = 'w'
        start_idx = 0
        _ = save_index_summary(name, rates, dates, grid_dim)
    elif os.path.exists(name + '.UNSMRY'):
        start_t = len(_read_sections(name + '.UNSMRY', decode=False, sequential=True)[1]['SEQHDR']) - 1
    else:
        logger_print("Dumping summary: No UNSMRY file found")

    save_summary_file(is_unified, rates, dates, name, mode, start_idx, start_t)
