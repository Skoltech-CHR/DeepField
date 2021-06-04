# pylint: skip-file
"""Configs collection."""
from .parse_utils.table_info import TABLE_INFO

orth_grid_config = {'OrthogonalUniformGrid': {'attrs': ['ACTNUM', 'DIMENS', 'DX', 'DY', 'DZ',
                                                        'MAPAXES', 'TOPS']}}
corn_grid_config = {'CornerPointGrid': {'attrs': ['ACTNUM', 'COORD', 'DIMENS', 'MAPAXES', 'ZCORN']}}
any_grid_config = {'Grid': {'attrs': list(set(orth_grid_config['OrthogonalUniformGrid']['attrs'] +
                                              corn_grid_config['CornerPointGrid']['attrs']))}}

base_config = {
    'Rock': {'attrs': ['PERMX', 'PERMY', 'PERMZ', 'PORO']},
    'States': {'attrs': ['PRESSURE', 'SOIL', 'SWAT', 'SGAS', 'RS']},
    'Tables': {'attrs': list(TABLE_INFO.keys())},
    'Wells': {'attrs': ['EVENTS', 'HISTORY', 'RESULTS', 'PERF', 'WELLTRACK',
                        'COMPDAT', 'WELSPECS', 'WCONPROD', 'WCONINJE']},
    'Aquifers': {'attrs': None}
}

default_orth_config = dict(orth_grid_config, **base_config)
default_corn_config = dict(corn_grid_config, **base_config)
default_config = dict(any_grid_config, **base_config)
