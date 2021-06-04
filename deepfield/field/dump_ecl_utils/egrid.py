"""EGRID dump assisting functions."""

import os
import datetime

from .share import BlockField, write_unrst_section, write_unrst_data_section_f, INTE, REAL

FILEHEAD = 'FILEHEAD'
GRIDHEAD = 'GRIDHEAD'
COORD = 'COORD'
ZCORN = 'ZCORN'
ACTNUM = 'ACTNUM'
ENDGRID = 'ENDGRID'

# Header's block formats
META_BLOCK_SPEC = {
    FILEHEAD: {
        'type': INTE,
        'items_cnt': 100,
        'struct': {
            'version': BlockField(idx=1, default=3),
            'release_year': BlockField(idx=2, default=lambda: datetime.datetime.today().year),
            'reserved_00': BlockField(idx=3, default=0),
            'backward_compatibility_version': BlockField(idx=4, default=0),
            'grid_type': BlockField(idx=5, default=0),            # 0 - corner point; 1 - unstructured; 2 - hybrid
            'dual_porosity_model': BlockField(idx=6, default=0),  # 0 - single porosity;
                                                                  # 1 - dual porosity;
                                                                  # 2 - dual permeability
            'original_grid_format': BlockField(idx=7, default=2)  # 0 - Unknown; 1 - Corner point; 2 - Block centered
        }
    },
    GRIDHEAD: {
        'type': INTE,
        'items_cnt': 100,
        'struct': {
            'grid_type': BlockField(idx=1, default=1),  # 0 - composite; 1 - corner point; 2 - unstructured
            'dimensions': BlockField(idx=2, cnt=3),
            'LGR_idx': BlockField(idx=5, default=0),     # 0 - global, >0 - LGR
            'numres': BlockField(idx=25, default=1),
            'nseg': BlockField(idx=26, default=1),
            'ntheta': BlockField(idx=27, default=0),
            'host_box': BlockField(idx=27, cnt=6, default=[1, 1, 1, 1, 1, 1])  # (lower i, j, k, upper i, j, k)
        }
    },
    ENDGRID: {
        'type': INTE,
        'items_cnt': 0
    }
}

DATA_BLOCK_SPEC = {
    COORD: REAL,
    ZCORN: REAL,
    ACTNUM: INTE
}

def save_egrid(field_grid, name, grid_dim, grid_format, mode):
    """
       Save ERGID target file

       Parameters
       ----------
       field_grid : np.array
       name : str
       grid_dim :

       Returns
       -------
    """
    if not (mode == 'w' or
            (mode == 'a' and not os.path.isfile(name + '.EGRID'))):
        return

    data = {}

    data['zcorn'] = field_grid.ravel(attr='ZCORN', inplace=False)
    data['coord'] = field_grid.ravel(attr='COORD', inplace=False)
    data['actnum'] = field_grid.ravel(attr='ACTNUM', inplace=False)

    with open(name + '.EGRID', 'w+b') as f:
        write_unrst_section(f, FILEHEAD, META_BLOCK_SPEC[FILEHEAD], grid_format=grid_format)
        write_unrst_section(f, GRIDHEAD, META_BLOCK_SPEC[GRIDHEAD], grid_dim)

        write_unrst_data_section_f(f=f, name=COORD, stype=DATA_BLOCK_SPEC[COORD],
                                   data_array=data[COORD.lower()])

        write_unrst_data_section_f(f=f, name=ZCORN, stype=DATA_BLOCK_SPEC[ZCORN],
                                   data_array=data[ZCORN.lower()])

        write_unrst_data_section_f(f=f, name=ACTNUM, stype=DATA_BLOCK_SPEC[ACTNUM],
                                   data_array=data[ACTNUM.lower()])

        write_unrst_section(f, ENDGRID, META_BLOCK_SPEC[ENDGRID])
