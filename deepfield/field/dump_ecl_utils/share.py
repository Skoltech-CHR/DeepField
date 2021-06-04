"""SHARE assisting functions."""
import math
import time
from datetime import datetime
from struct import pack, unpack
from math import fmod

import numpy as np

# Spec section
TIME = 'TIME'
ITIME = 'ITIME'
NAME = 'NAME'
TYPE = 'TYPE'
NUMBER = 'NUMBER'
POINTER = 'POINTER'
POINTERB = 'POINTERB'
ARRAYMAX = 'ARRAYMAX'
ARRAYMIN = 'ARRAYMIN'
LGRSGONE = 'LGRSGONE'

# Types
INTE = 'INTE'
REAL = 'REAL'
LOGI = 'LOGI'
DOUB = 'DOUB'
CHAR = 'CHAR'
MESS = 'MESS'
C008 = 'C008'
NOTYPE = '----'


SECTION_TPL = ['16', '"{name: <8}".format(name=name)',
               'items_cnt', '"{type: <4}".format(type=stype)',
               '16', 'data_size', 'data', 'data_size']
SECTION_TPL_EMPTY = ['16', '"{name: <8}".format(name=name)',
                     'items_cnt', '"{type: <4}".format(type=stype)', '16']
SECTION_TPL_DATA = ['16', '"{name: <8}".format(name=name)', 'items_cnt',
                    '"{type: <4}".format(type=stype)', '16', 'data']


POINTER_MAX = 2**31

def get_filepath_wo_ext(file_spec):
    """
    Get file path without extension

    Parameters
    ----------
    file_spec : DataStruct
        The function use attributes output_dir and fname_wo_ext for construct file path

    Returns
    -------
    out : str
        Constructed file path

    """
    return file_spec.output_dir + '/' + file_spec.fname_wo_ext


class SpecElement:
    """SpecElement struct.

        Parameters
        ----------
    """
    def __init__(self, name=None, el_type=None, number=None, el_struct=None):
        if el_struct:
            self.name = el_struct['name']
            self.type = el_struct['type']
            self.number = el_struct['items_cnt']
        else:
            self.name = name
            self.type = el_type
            self.number = number


class BlockField:
    """BlockField.

        Parameters
        ----------
    """
    def __init__(self, idx, cnt=1, default: object = 0x00):
        self.index = idx
        self.el_count = cnt
        if callable(default):
            self.val = default
        else:
            self.val = lambda: default


class DataType():
    """DataType.

        Parameters
        ----------
    """
    def __init__(self, el_size, el_format, el_skip, el_type, endian=''):
        self.el_size = el_size
        self.el_format = el_format
        self.el_skip = el_skip
        self.el_type = el_type
        self.endian = endian


DATA_TYPES = {
    INTE: DataType(4, 'i', 1000, int, '>'),
    REAL: DataType(4, 'f', 1000, float, '>'),
    LOGI: DataType(4, 'i', 1000, int, '>'),
    DOUB: DataType(8, 'd', 1000, np.float64, '>'),
    CHAR: DataType(8, '8s', 105, str, '>'),
    MESS: DataType(8, '8s', 105, str, '>'),
    C008: DataType(8, '8s', 105, str, '>'),
    NOTYPE: DataType(8, '8s', 105, str, '>')
}

INTEHEAD_ITEMS_CNT = 0x0128
LOGIHEAD_ITEMS_CNT = 0x0064
DOUBHEAD_ITEMS_CNT = 0x00D9
TIME_ITEMS_CNT = 1
ITIME_ITEMS_CNT = 13

SPEC_DATA_BLOC_SPEC = {
    TIME: DOUB,
    ITIME: INTE,
    NAME: CHAR,
    TYPE: CHAR,
    NUMBER: INTE,
    POINTER: INTE,
    POINTERB: INTE,
    ARRAYMAX: INTE,
    ARRAYMIN: INTE
}

INTEHEAD = 'INTEHEAD'
LOGIHEAD = 'LOGIHEAD'
DOUBHEAD = 'DOUBHEAD'
IGRP = 'IGRP'
IWEL = 'IWEL'
ZWEL = 'ZWEL'
ICON = 'ICON'
STARTSOL = 'STARTSOL'
ENDSOL = 'ENDSOL'

META_BLOCK_SPEC = {
    INTEHEAD: {
        'name': INTEHEAD,
        'type': INTE,
        'items_cnt': INTEHEAD_ITEMS_CNT,
        'struct': {
            'creation_time': BlockField(idx=1, default=lambda: int(time.time())),
            'units_type': BlockField(idx=3, default=1),  # 1-metric, 2-field, 3-lab, 4-PVT-M
            'dimensions': BlockField(idx=9, cnt=3),
            'n_active_cells': BlockField(idx=12),
            'grid_type': BlockField(idx=14),
            'i_phase': BlockField(idx=15, default=7),
            # [1-oil, 2-water, 3-oil/water, 4-gas, 5-oil/gas. 6-gas/water, 7-oil/water/gas]
            'n_wells': BlockField(idx=17),  # ???
            'n_max_completitions_per_well': BlockField(idx=18),  # ???
            'n_max_wells_per_group': BlockField(idx=20),  # ???
            'n_max_groups': BlockField(idx=21),  # ???
            'n_data_per_well': BlockField(idx=25),  # in IWELL array  #???
            'n_words_per_well': BlockField(idx=28),  # n of 8-char words in ZWELL array  #???
            'n_data_per_completition': BlockField(idx=33),  # in ICON array  #???
            'n_data_per_group': BlockField(idx=37),  # in IGRP array  #???
            'date': BlockField(idx=65, cnt=3,
                               default=lambda: [datetime.today().day, datetime.today().month,
                                                datetime.today().year]), # date of the report time
            'program_id': BlockField(idx=95, default=-1),
            'n_max_segmented_wells': BlockField(idx=176),  # ???
            'n_max_segments_per_well': BlockField(idx=177),  # ???
            'n_data_per segment': BlockField(idx=179),  # in ISEG array  #???
        }
    },
    LOGIHEAD: {
        'name': LOGIHEAD,
        'type': LOGI,
        'items_cnt': LOGIHEAD_ITEMS_CNT,
        'struct': {
            'radial_model_flag_300': BlockField(idx=4),
            'radial_model_flag_100': BlockField(idx=5),
            'dual_porosity_flag': BlockField(idx=15),
            'coal_bed_methane_flag': BlockField(idx=31)
        }
    },
    DOUBHEAD: {
        'name': DOUBHEAD,
        'type': DOUB,
        'items_cnt': DOUBHEAD_ITEMS_CNT,
        'struct': {
            'time_in_days': BlockField(idx=1)
        }
    },
    LGRSGONE: {
        'name': LGRSGONE,
        'type': MESS,
        'items_cnt': 0
    },
    IGRP: {
        'name': IGRP,
        'type': INTE,
        'items_cnt': 0
    },
    IWEL: {
        'name': IWEL,
        'type': INTE,
        'items_cnt': 0
    },
    ZWEL: {
        'name': ZWEL,
        'type': CHAR,
        'items_cnt': 0
    },
    ICON: {
        'name': ICON,
        'type': INTE,
        'items_cnt': 0
    },
    STARTSOL: {
        'name': STARTSOL,
        'type': MESS,
        'items_cnt': 0
    },
    ENDSOL: {
        'name': ENDSOL,
        'type': MESS,
        'items_cnt': 0
    }
}


def split_pointer(pointer_in):
    """
        split_pointer

        Parameters
        ----------
        pointer_in :

        Returns
        -------
        [pointer_a, pointer_b] : list

    """
    pointer_a = int(fmod(pointer_in, POINTER_MAX))
    pointer_b = int(pointer_in/POINTER_MAX)
    return [pointer_a, pointer_b]


def format_keyword(name):
    """
    Get line alignment by 8 chars length

    Parameters
    ----------
    name : str
        String for alignment

    Returns
    -------
        out : str
            8 chars length string

    """
    return "{name: <8}".format(name=name.upper())


def write_unrst_section(f, name, spec, grid_dim=None, nactive=None, units_type=None,
                        grid_type=None, sdate=None, i_phase=None, grid_format=None):
    """
    Write metadata section into destination file

    Parameters
    ----------
    f : file object
        target file
    name : str
        section name
    spec : dict
        section specification
    grid_dim : (int, int, int)
        grid dimension (x, y, z)

    """
    stype = spec['type']
    items_cnt = spec['items_cnt']
    data_type = DATA_TYPES[stype]
    writen_size = 0
    if items_cnt > 0:
        data_size = items_cnt * data_type.el_size
        data_array = np.zeros(items_cnt + 1, dtype=data_type.el_type)
        for key in spec['struct'].keys():
            field_spec = spec['struct'][key]
            if key == 'dimensions':
                val = grid_dim
            elif (key == 'units_type') and (not units_type is None):
                val = units_type
            elif key == 'n_active_cells':
                if not nactive is None:
                    val = nactive
                else:
                    val = grid_dim[0]*grid_dim[1]*grid_dim[2]
            elif (key == 'grid_type') and (not grid_type is None):
                val = grid_type
            elif (key == 'date') and (not sdate is None):
                val = [sdate.day, sdate.month, sdate.year]
            elif (key == 'i_phase') and (not i_phase is None):
                val = i_phase
            elif key == 'nlist':
                val = field_spec.val
            elif key == 'original_grid_format':
                val = grid_format
            else:
                val = field_spec.val()
            if field_spec.el_count > 1:
                for i in range(field_spec.el_count):
                    data_array[field_spec.index+i] = val[i]
            else:
                data_array[field_spec.index] = val
        data_format = data_type.endian + (data_type.el_format * items_cnt)
        data = pack(data_format, *data_array[1:])

    writen_size += f.write(pack('>i', 16))
    writen_size += f.write("{name: <8}".format(name=name).encode('ascii'))
    writen_size += f.write(pack('>i', items_cnt))
    writen_size += f.write("{type: <4}".format(type=stype).encode('ascii'))
    writen_size += f.write(pack('>i', 16))
    if items_cnt > 0:
        writen_size += f.write(pack('>i', data_size))
        writen_size += f.write(data)
        writen_size += f.write(pack('>i', data_size))

    return writen_size


def if_string_then_convert_to_bytes(val):
    """
    Convert string to bytes array

    Parameters
    ----------
    val : str or bytes

    Returns
    -------
    out : bytes

    """
    if isinstance(val, str):
        return bytearray(val, 'ascii')

    return val


def write_unrst_data_section(f, name, stype, data_array):
    """
        Write data section into destination file

        Parameters
        ----------
        f : file object
            target file
        name : str
            section name
        stype : str
            data elements' type
        data_array : np.array(int, int, int)
            grid dimension (x, y, z)

    """
    writen_size = 0
    data_type = DATA_TYPES[stype]

    _data0 = data_array.reshape(-1).tolist()
    _data1 = list(map(lambda v: if_string_then_convert_to_bytes(v), _data0))

    items_cnt = len(_data1)
    subsections_num = math.floor(items_cnt / data_type.el_skip)
    last_subsect_items = items_cnt % data_type.el_skip
    data_format = data_type.endian
    _data2 = []
    subsect_size = data_type.el_skip * data_type.el_size
    for i in range(subsections_num):
        data_format += 'i' + (data_type.el_format * data_type.el_skip) + 'i'
        _data2.append(subsect_size)
        _data2.extend(_data1[i * data_type.el_skip:(i+1) * data_type.el_skip])
        _data2.append(subsect_size)
    data_format += 'i' + (data_type.el_format * last_subsect_items) + 'i'
    _data2.append(last_subsect_items * data_type.el_size)
    _data2.extend(_data1[subsections_num * data_type.el_skip:])
    _data2.append(last_subsect_items * data_type.el_size)
    data = pack(data_format, *_data2)

    writen_size += f.write(pack('>i', 16))
    writen_size += f.write("{name: <8}".format(name=name).encode('ascii'))
    writen_size += f.write(pack('>i', items_cnt))
    writen_size += f.write("{type: <4}".format(type=stype).encode('ascii'))
    writen_size += f.write(pack('>i', 16))
    writen_size += f.write(data)

    return writen_size

def write_unrst_data_section_f(f, name, stype, data_array):
    """
        Write data section into destination file fast but ints and floats only

        Parameters
        ----------
        f : file object
            target file
        name : str
            section name
        stype : str
            data elements' type
        data_array : np.array(int, int, int)
            grid dimension (x, y, z)

    """
    writen_size = 0

    data_type = DATA_TYPES[stype]

    array_type = '>' + data_type.el_format

    # if stype == 'CHAR':
    #     data = data_array.reshape(-1).astype(np.bytes_)
    # else:
    data = data_array.astype(array_type)

    items_cnt = len(data)

    subsections_num = math.floor(items_cnt / data_type.el_skip)
    last_subsect_items = items_cnt % data_type.el_skip

    subsect_size = data_type.el_skip * data_type.el_size

    if last_subsect_items != 0:
        new_data = np.zeros(items_cnt + 2*subsections_num + 2, dtype=array_type)
    else:
        new_data = np.zeros(items_cnt + 2*subsections_num, dtype=array_type)

    for i in range(subsections_num):
        new_data[i*(data_type.el_skip + 2)] = unpack(array_type, pack(">i", subsect_size))[0]
        new_data[i*(data_type.el_skip + 2) + 1:i*(data_type.el_skip + 2) + 1 + data_type.el_skip] = \
            data[i*data_type.el_skip:(i+1)*data_type.el_skip]
        new_data[(i+1)*(data_type.el_skip + 2) - 1] = unpack(array_type, pack(">i", subsect_size))[0]

    if last_subsect_items != 0:
        new_data[subsections_num*(data_type.el_skip + 2)] = \
            unpack(array_type, pack(">i", (last_subsect_items * data_type.el_size)))[0]
        new_data[subsections_num*(data_type.el_skip + 2) + 1:subsections_num*(data_type.el_skip + 2) \
                                                             + 1 + last_subsect_items] = \
            data[subsections_num*data_type.el_skip:]
        new_data[(subsections_num)*(data_type.el_skip + 2) + 1 + last_subsect_items] = \
            unpack(array_type, pack(">i", (last_subsect_items * data_type.el_size)))[0]

    data = new_data.tobytes()

    #data = pack(data_format, *new_data)

    writen_size += f.write(pack('>i', 16))
    writen_size += f.write("{name: <8}".format(name=name).encode('ascii'))
    writen_size += f.write(pack('>i', items_cnt))
    writen_size += f.write("{type: <4}".format(type=stype).encode('ascii'))
    writen_size += f.write(pack('>i', 16))
    writen_size += f.write(data)

    return writen_size
