"""Dump tools."""
import numpy as np
import pandas as pd

PERF_VALUE_COLUMNS = ['RAD', 'DIAM', 'SKIN', 'MULT']

def write_perf(f, wells, defaults):
    """Write perforations to file."""
    dfs = []
    for node in wells:
        if 'PERF' in node.attributes:
            if node.perf.empty:
                continue
            df = node.perf.copy()
            df['WELL'] = node.name
            dfs.append(df)
    if dfs:
        df = pd.concat(dfs, sort=False)
    else:
        return

    if 'COVERED' in df: #TODO: this should not happen at all
        df.drop('COVERED', axis=1, inplace=True)

    f.write('EFORm WELL \'DD.MM.YYYY\' MDL MDU ' +
            ' '.join([c for c in df if c in PERF_VALUE_COLUMNS]) + '\n')
    f.write('ETAB\n')

    for c in df.columns:
        if c in defaults:
            df[c] = df[c].fillna(defaults[c])
    if df.isna().any().any():
        raise ValueError('Perforations contains nans.')

    df['PERF'] = 'PERF'
    df['BRANCH'] = df['WELL'].str.split(':').apply(lambda x: 'BRANCH {}'.format(':'.join(x[1:]))
                                                   if len(x) > 1 else '')
    df['WELL'] = df['WELL'].str.split(':').apply(lambda x: x[0])
    df['DATE'] = df['DATE'].dt.strftime('%d.%m.%Y')
    if 'CLOSE' in df:
        df['CLOSE'] = df['CLOSE'].apply(lambda x: 'CLOSE' if x else '')

    df = df[['WELL', 'DATE', 'PERF', 'MDL', 'MDU'] +
            [c for c in df if c in PERF_VALUE_COLUMNS] +
            ['BRANCH'] +
            (['CLOSE'] if 'CLOSE' in df else [])]

    f.write(df.to_string(header=False, index=False, index_names=False) + '\n')
    f.write('ENDE\n')

def expand_event_df(df, value_control_kw):
    """Add control keywords to columns."""
    order = []
    for col in df.columns:
        if col in value_control_kw:
            df[col + '_'] = col
            order.extend([col + '_', col])
        elif col == 'MODE':
            order = ['MODE'] + order
    order = ['WELL', 'DATE'] + order
    return df[order]

def write_events(f, wells, value_control_kw):
    """Write perforations to file."""
    dfs = []
    for node in wells:
        if 'EVENTS' in node.attributes:
            if node.events.empty:
                continue
            df = node.events.copy()
            df['WELL'] = node.name
            dfs.append(df)
    if dfs:
        df = pd.concat(dfs, sort=False)
    else:
        return

    f.write('EFORm WELL \'DD.MM.YYYY\'\n')
    f.write('ETAB\n')

    df = df[['WELL'] + [col for col in df if col != 'WELL']]

    df['DATE'] = df['DATE'].dt.strftime('%d.%m.%Y')

    if 'MODE' in df:
        for _, df_mode in df.groupby('MODE'):
            df_mode = df_mode.dropna(axis=1)
            df_mode = expand_event_df(df_mode, value_control_kw)
            f.write(df_mode.to_string(header=False, index=False, index_names=False) + '\n')
    else:
        df = df.dropna(axis=1)
        f.write(df_mode.to_string(header=False, index=False, index_names=False) + '\n')
    f.write('ENDE\n')

def write_schedule(f, wells):
    """Write SCHEDULE file."""
    dfc = []
    dfp = []
    dfi = []
    for node in wells:
        if 'COMPDAT' in node.attributes:
            dfc.append(node.compdat)
        if 'WCONPROD' in node.attributes:
            dfp.append(node.wconprod)
        if 'WCONINJE' in node.attributes:
            dfi.append(node.wconinje)
    if not dfc:
        return

    dfc = pd.concat(dfc, sort=False)
    dfp = pd.concat(dfp, sort=False) if dfp else pd.DataFrame(columns=['DATE'])
    dfi = pd.concat(dfi, sort=False) if dfi else pd.DataFrame(columns=['DATE'])

    for df in [dfc, dfp, dfi]:
        df['END_LINE'] = '/'

    def write_group(group):
        group = group.drop('DATE', axis=1).fillna('1*')
        f.write(group.to_string(header=False, index=False, index_names=False) + '\n')
        f.write('/\n\n')

    group = dfc.loc[dfc['DATE'].isna()]
    if not group.empty:
        f.write('COMPDAT\n')
        write_group(group)

    group = dfp.loc[dfp['DATE'].isna()]
    if not group.empty:
        f.write('WCONPROD\n')
        write_group(group)

    group = dfi.loc[dfi['DATE'].isna()]
    if not group.empty:
        f.write('WCONINJE\n')
        write_group(group)

    dates = (list(dfc['DATE'].dropna()) +
             list(dfp['DATE'].dropna()) +
             list(dfi['DATE'].dropna()))
    dates = np.unique(dates)
    dates.sort()

    for date in dates:
        if date == dates[0]:
            f.write('DATES\n{} 00:00:00.001 /\n/\n\n'
                    .format(date.strftime('%d %b %Y')).upper())
        else:
            f.write('DATES\n{} /\n/\n\n'.format(date.strftime('%d %b %Y')).upper())
        group = dfc.loc[dfc['DATE'].values == date]
        if not group.empty:
            f.write('COMPDAT\n')
            write_group(group)

        group = dfp.loc[dfp['DATE'].values == date]
        if not group.empty:
            f.write('WCONPROD\n')
            write_group(group)

        group = dfi.loc[dfi['DATE'].values == date]
        if not group.empty:
            f.write('WCONINJE\n')
            write_group(group)

def write_welspecs(f, wells):
    """Write WELSPECS to file."""
    dfs = []
    for node in wells:
        if 'WELSPECS' in node.attributes and not node.welspecs.empty:
            dfs.append(node.welspecs)
    if not dfs:
        return

    df = pd.concat(dfs, sort=False).sort_values('WELL')
    df['END_LINE'] = '/'
    df[['I', 'J']] = df[['I', 'J']].astype(int)
    df = df.fillna('1*')

    f.write('WELSPECS\n')
    f.write(df.to_string(header=False, index=False, index_names=False) + '\n')
    f.write('/\n\n')
