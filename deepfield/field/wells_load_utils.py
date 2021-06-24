"""Wells utils."""
import re
import numpy as np
import pandas as pd

from .well_segment import WellSegment
from .utils import get_single_path
from .parse_utils import (read_rsm, parse_perf_line, parse_control_line,
                          parse_history_line, read_ecl_bin)

DEFAULTS = {'RAD': 0.1524, 'DIAM': 0.3048, 'SKIN': 0, 'MULT': 1, 'CLOSE': False,
            'MODE': 'OPEN', 'DIR': 'Z', 'GROUP': 'FIELD'}

MODE_CONTROL = ['PROD', 'INJE', 'STOP']
VALUE_CONTROL = ['BHPT', 'THPT', 'DRAW', 'ETAB', 'OPT', 'GPT', 'WPT', 'LPT', 'VPT',
                 'OIT', 'GIT', 'WIT',
                 'HOIL', 'HGAS', 'HWAT', 'HLIQ', 'HBHP', 'HTHP', 'HWEF',
                 'GOPT', 'GGPT', 'GWPT', 'GLPT',
                 'GGIT', 'GWIT',
                 'OIL', 'GAS', 'WAT', 'LIQ', 'BHP', 'THP', 'GOR', 'OGR', 'WCT', 'WOR', 'WGR',
                 'DREF'
                 ]

def load_rsm(wells, path, logger):
    """Load RSM well data from file."""
    logger.info("Start reading {}".format(path))
    rsm = read_rsm(path, logger)
    logger.info("Finish reading {}".format(path))
    if '_children' in rsm['_global']:
        del rsm['_global']['_children']
    df = pd.DataFrame({k: v['data'] for k, v in rsm['_global'].items()})
    dates = pd.to_datetime(df[['YEAR', 'MONTH', 'DAY']])
    welldata = {}
    for wellname, data in rsm.items():
        if wellname == '_global':
            continue
        if '_children' in data:
            del data['_children']
        wellname = wellname.strip(' \t\'\"').upper()
        wdf = pd.DataFrame({k: v['data'] * v['multiplyer'] for k, v in data.items()})
        wdf['DATE'] = dates
        wdf = wdf[['DATE'] + [col for col in wdf.columns if col != 'DATE']]
        welldata[wellname] = {'RESULTS': wdf.sort_values('DATE')}
    return wells.update(welldata)

def load_ecl_binary(wells, path_to_results, attrs, basename, logger=None, **kwargs):
    """Load results from UNSMRY file."""
    _ = kwargs

    if 'RESULTS' not in attrs:
        return wells

    smry_path = get_single_path(path_to_results, basename + '.UNSMRY', logger)
    if smry_path is None:
        return wells

    spec_path = get_single_path(path_to_results, basename + '.SMSPEC', logger)
    if spec_path is None:
        return wells

    def is_well_name(s):
        return re.match(r"[a-zA-Z0-9]", s) is not None

    smry_data = np.stack(read_ecl_bin(smry_path, attrs=['PARAMS'],
                                      sequential=True, logger=logger)['PARAMS'])
    spec_dict = read_ecl_bin(spec_path, attrs=['KEYWORDS', 'WGNAMES'],
                             sequential=False, logger=logger)
    kw = [w.strip() for w in spec_dict['KEYWORDS']]
    wellnames = [w.strip() for w in spec_dict['WGNAMES']]

    df = pd.DataFrame({k: smry_data[:, kw.index(k)].astype(int)
                       for k in ['DAY', 'MONTH', "YEAR"]})
    dates = pd.to_datetime(df.YEAR*10000 + df.MONTH*100 + df.DAY, format='%Y%m%d')

    welldata = {w: {'RESULTS': pd.DataFrame({'DATE': dates})}
                for w in np.unique(wellnames) if is_well_name(w)}
    for i, w in enumerate(wellnames):
        if w not in welldata:
            continue
        welldata[w]['RESULTS'][kw[i]] = smry_data[:, i]
    for v in welldata.values():
        v['RESULTS'].sort_values('DATE', inplace=True)
    return wells.update(welldata)

def load_group(wells, buffer, **kwargs):
    """Load groups. Note: optional keyword FRAC is not implemented."""
    _ = kwargs
    group = next(iter(buffer)).upper().split('FRAC')[0].split()
    group_name = group[1]
    if group_name == '1*':
        group_name = DEFAULTS['GROUP']
    try:
        group_node = wells[group_name]
    except KeyError:
        group_node = WellSegment(parent=wells.root, name=group_name, is_group=True)
    for well in group[2:]:
        try:
            node = wells[well]
            node.parent = group_node
        except KeyError:
            WellSegment(parent=group_node, name=well)
    return wells

def load_grouptree(wells, buffer, **kwargs):
    """Load grouptree."""
    _ = kwargs
    for line in buffer:
        if line.strip() == '/':
            return wells
        node, grp = re.sub("[\"\']", "", line).split('/')[0].strip().split()
        if grp == '1*':
            grp = DEFAULTS['GROUP']
        try:
            node = wells[node]
        except KeyError:
            node = WellSegment(parent=wells.root, name=node, is_group=True)
        try:
            grp = wells[grp]
        except KeyError:
            grp = WellSegment(parent=wells.root, name=grp, is_group=True)
        node.parent = grp
    return wells

def load_welspecs(wells, buffer, **kwargs):
    """Partial load WELSPECS table."""
    _ = kwargs
    columns = ['WELL', 'GROUP', 'I', 'J', 'DREF', 'PHASE']
    df = pd.DataFrame(columns=columns)
    for line in buffer:
        if '/' not in line:
            break
        line = line.split('/')[0].strip()
        if not line:
            break
        vals = line.split()[:len(columns)]
        full = [None] * len(columns)
        shift = 0
        for i, v in enumerate(vals):
            if i + shift >= len(columns):
                break
            if '*' in v:
                shift += int(v.strip('*')) - 1
            else:
                full[i+shift] = v
        df = df.append(dict(zip(columns, full)), ignore_index=True)
    df[['WELL', 'GROUP', 'PHASE']] = df[['WELL', 'GROUP', 'PHASE']].applymap(
        lambda x: x.strip('\'\"') if x is not None else x)
    df[['I', 'J']] = df[['I', 'J']].astype(int, errors='ignore')
    df['DREF'] = df['DREF'].astype(float, errors='ignore')
    for k, v in DEFAULTS.items():
        if k in df:
            df[k] = df[k].fillna(v)
    if not df.empty:
        welldata = {k: {'WELSPECS': v.reset_index(drop=True)} for k, v in df.groupby('WELL')}
        wells.update(welldata, mode='a', ignore_index=True)
    return wells

def load_wconprod(wells, buffer, meta, **kwargs):
    """Partial load WCONPROD table."""
    _ = kwargs
    dates = meta['DATES']
    columns = ['DATE', 'WELL', 'MODE', 'CONTROL',
               'OPT', 'WPT', 'GPT', 'SLPT', 'LPT', 'BHPT']
    df = pd.DataFrame(columns=columns)
    for line in buffer:
        if '/' not in line:
            break
        line = line.split('/')[0].strip()
        if not line:
            break
        vals = line.split()[:len(columns)]
        full = [None] * len(columns)
        full[0] = dates[-1] if not dates.empty else pd.to_datetime('')
        shift = 1
        for i, v in enumerate(vals):
            if i + shift >= len(columns):
                break
            if '*' in v:
                shift += int(v.strip('*')) - 1
            else:
                full[i+shift] = v
        df = df.append(dict(zip(columns, full)), ignore_index=True)
    df[['WELL', 'MODE', 'CONTROL']] = df[['WELL', 'MODE', 'CONTROL']].applymap(
        lambda x: x.strip('\'\"') if x is not None else x)
    df[df.columns[4:]] = df[df.columns[4:]].astype(float, errors='ignore')
    for k, v in DEFAULTS.items():
        if k in df:
            df[k] = df[k].fillna(v)
    if not df.empty:
        welldata = {k: {'WCONPROD': v.reset_index(drop=True)} for k, v in df.groupby('WELL')}
        wells.update(welldata, mode='a', ignore_index=True)
    return wells

def load_wconinje(wells, buffer, meta, **kwargs):
    """Partial load WCONINJE table."""
    _ = kwargs
    dates = meta['DATES']
    columns = ['DATE', 'WELL', 'PHASE', 'MODE', 'CONTROL', 'SPIT', 'PIT', 'BHPT']
    df = pd.DataFrame(columns=columns)
    for line in buffer:
        if '/' not in line:
            break
        line = line.split('/')[0].strip()
        if not line:
            break
        vals = line.split()[:len(columns)]
        full = [None] * len(columns)
        full[0] = dates[-1] if not dates.empty else pd.to_datetime('')
        shift = 1
        for i, v in enumerate(vals):
            if i + shift >= len(columns):
                break
            if '*' in v:
                shift += int(v.strip('*')) - 1
            else:
                full[i+shift] = v
        df = df.append(dict(zip(columns, full)), ignore_index=True)
    df[df.columns[1:5]] = df[df.columns[1:5]].applymap(
        lambda x: x.strip('\'\"') if x is not None else x)
    df[df.columns[5:]] = df[df.columns[5:]].astype(float, errors='ignore')
    for k, v in DEFAULTS.items():
        if k in df:
            df[k] = df[k].fillna(v)
    if not df.empty:
        welldata = {k: {'WCONINJE': v.reset_index(drop=True)} for k, v in df.groupby('WELL')}
        wells.update(welldata, mode='a', ignore_index=True)
    return wells

def load_compdat(wells, buffer, meta, **kwargs):
    """Load COMPDAT table."""
    _ = kwargs
    dates = meta['DATES']
    columns = ['DATE', 'WELL', 'I', 'J', 'K1', 'K2', 'MODE', 'Sat',
               'CF', 'DIAM', 'KH', 'SKIN', 'ND', 'DIR', 'Ro']
    df = pd.DataFrame(columns=columns)
    for line in buffer:
        if '/' not in line:
            break
        line = line.split('/')[0].strip()
        if not line:
            break
        vals = line.split()
        full = [None] * len(columns)
        full[0] = dates[-1] if not dates.empty else pd.to_datetime('')
        shift = 1
        for i, v in enumerate(vals):
            if '*' in v:
                shift += int(v.strip('*')) - 1
            else:
                full[i+shift] = v
        df = df.append(dict(zip(columns, full)), ignore_index=True)
    df[['WELL', 'MODE', 'DIR']] = df[['WELL', 'MODE', 'DIR']].applymap(
        lambda x: x.strip('\'\"') if x is not None else x)
    df[['I', 'J', 'K1', 'K2']] = df[['I', 'J', 'K1', 'K2']].astype(int)
    df[['Sat', 'CF', 'DIAM', 'KH', 'Ro']] = df[['Sat', 'CF', 'DIAM', 'KH', 'Ro']].astype(float)
    for k, v in DEFAULTS.items():
        if k in df:
            df[k] = df[k].fillna(v)
    if not df.empty:
        welldata = {k: {'COMPDAT': v.reset_index(drop=True)} for k, v in df.groupby('WELL')}
        wells.update(welldata, mode='a', ignore_index=True)
    return wells

def load_welltracks(wells, buffer, **kwargs):
    """Load welltracks while possible.

    Parameters
    ----------
    buffer : StringIteratorIO
        Buffer to get string from.

    Returns
    -------
    comp : Wells
        Wells component with loaded well data.
    """
    _ = kwargs
    welldata = {}
    while 1:
        track = get_single_welltrack(buffer)
        if track:
            welldata.update(track)
        else:
            return wells.update(welldata)

def get_single_welltrack(buffer):
    """Load single welltrack."""
    track = []
    try:
        line = next(buffer)
    except StopIteration:
        return {}
    line = ' '.join([word for word in line.split() if word.upper() not in ['WELLTRACK']])
    name = line.strip(' \t\'\"').upper()
    last_line = False
    for line in buffer:
        if '/' in line:
            line = line.split('/')[0]
            last_line = True
        try:
            p = np.array(line.split(maxsplit=4)[:4]).astype(np.float)
            assert len(p) == 4
            track.append(p)
        except (ValueError, AssertionError):
            buffer.prev()
            break
        if last_line:
            break
    if track:
        return {name: {"WELLTRACK": np.array(track)}}
    return {}

def load_events(wells, buffer, column_names, logger, **kwargs):
    """Load perforations and events from event table."""
    _ = kwargs
    column_names = [s.upper() for s in column_names]
    if column_names[0] != 'WELL':
        logger.info("Expected WELL in the first column, found {}.".format(column_names[0]))
        return wells
    if column_names[1].strip('\'\"') != "DD.MM.YYYY":
        logger.info("Expected 'DD.MM.YYYY' in the second column, found {}.".format(column_names[1]))
        return wells
    column_names[1] = 'DATE'

    df_perf = pd.DataFrame()
    df_evnt = pd.DataFrame()

    for line in buffer:
        if 'ENDE' in line or line.strip() == '/':
            break
        if 'PERF' in line.upper():
            df_perf = df_perf.append(parse_perf_line(line, column_names, DEFAULTS))
        elif 'SQUE' in line.upper():
            continue
        else:
            df_evnt = df_evnt.append(parse_control_line(line, MODE_CONTROL, VALUE_CONTROL))

    if not df_perf.empty:
        welldata = {k: {'PERF': v.reset_index(drop=True).sort_values('DATE')}
                    for k, v in df_perf.groupby('WELL')}
        wells.update(welldata, mode='a', ignore_index=True)

    if not df_evnt.empty:
        welldata = {k: {'EVENTS': v.reset_index(drop=True).sort_values('DATE')}
                    for k, v in df_evnt.groupby('WELL')}
        wells.update(welldata, mode='a', ignore_index=True)
    return wells

def load_history(wells, buffer, column_names, logger, **kwargs):
    """Load history rates."""
    _ = kwargs
    column_names = [s.upper() for s in column_names]
    if column_names[0] != 'WELL':
        logger.info("Expected WELL in a first column, found {}.".format(column_names[0]))
        return wells
    if column_names[1].strip('\'\"') != "DD.MM.YYYY":
        logger.info("Expected 'DD.MM.YYYY' in a second column, found {}.".format(column_names[1]))
        return wells
    column_names[1] = 'DATE'

    df = pd.DataFrame()

    for line in buffer:
        if 'ENDE' in line or 'ENDH' in line or line.strip() == '/':
            break
        df = df.append(parse_history_line(line, column_names))

    if not df.empty:
        welldata = {k: {'HISTORY': v.reset_index(drop=True)} for k, v in df.groupby('WELL')}
        wells.update(welldata, mode='a', ignore_index=True)
    return wells
