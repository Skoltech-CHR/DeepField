"""INIT dump assisting functions."""

import os
import numpy as np

from .share import INTEHEAD, LOGIHEAD, DOUBHEAD, SpecElement, META_BLOCK_SPEC, REAL, \
    write_unrst_section, NAME, SPEC_DATA_BLOC_SPEC, TYPE, NUMBER, POINTER, POINTERB, ARRAYMAX, \
    ARRAYMIN, format_keyword, write_unrst_data_section, write_unrst_data_section_f

PORO = 'PORO'
PERMX = 'PERMX'
PERMY = 'PERMY'
PERMZ = 'PERMZ'

# Header's block formats
INDEX_SECTIONS_DATA = {
    INTEHEAD: SpecElement(el_struct=META_BLOCK_SPEC[INTEHEAD]),
    LOGIHEAD: SpecElement(el_struct=META_BLOCK_SPEC[LOGIHEAD]),
    DOUBHEAD: SpecElement(el_struct=META_BLOCK_SPEC[DOUBHEAD]),
    PORO: SpecElement(PORO, REAL),
    PERMX: SpecElement(PERMX, REAL),
    PERMY: SpecElement(PERMY, REAL),
    PERMZ: SpecElement(PERMZ, REAL)
}

DATA_BLOCK_SPEC = {
    PORO: SpecElement(PORO, REAL),
    PERMX: SpecElement(PERMX, REAL),
    PERMY: SpecElement(PERMY, REAL),
    PERMZ: SpecElement(PERMZ, REAL)
}

def get_type_section_data(key_list):
    """
    Get type list for TYPE section in INSPEC target file

    Parameters
    ----------
    key_list : list
        The list of keys for which you want to create a list of types

    Returns
    -------
    out : np.array
        Type list for given keys

    See Also
    --------
    np.array

    Examples
    --------
    >>> l = get_type_section_data(['INTEHEAD', 'LOGIHEAD'])
    >>> l
    array(['INTE    ', 'LOGI    '], dtype='<U8')

    """
    return np.array(list(map(lambda key: format_keyword(INDEX_SECTIONS_DATA[key].type), key_list)))


def get_number_section_data(key_list):
    """
    Get number list of items for target sections

    Parameters
    ----------
    key_list : list
        The list of keys for which you want to get number of items

    Returns
    -------
    out : np.array
        Number list of items in target sections

    See Also
    --------
    np.array

    Examples
    --------
    >>> l = get_number_section_data(['INTEHEAD', 'LOGIHEAD'])
    >>> l
    array([296, 100])

    """
    return np.array(list(map(lambda key: INDEX_SECTIONS_DATA[key].number, key_list)))


def get_pointer_section_data(key_list, pointers):
    """
    Get offsets if target sections in INIT file

    Parameters
    ----------
    key_list : list
        List of keys for which you want to get number of items
    pointers : PointerStruct
        List of offsets of the specified sections in target INIT file

    Returns
    -------
    out : np.array
        List of offsets of offsets of target sections

    See Also
    --------
    np.array

    Examples
    --------
    >>> l = get_pointer_section_data(['INTEHEAD', 'LOGIHEAD'], pointers)
    >>> l
    array([43, 432])

    """
    return np.array(list(map(lambda key: pointers.ptrs[key], key_list)))


def get_arraymax_section_data(key_list, max_size=1300000000):
    """
    List of array max parameter of target sections in INIT file

    Parameters
    ----------
    key_list : list
        List of keys for which you want to get number of items
    max_size : int, optional
        Max size, default 1300000000

    Returns
    -------
    out : np.array
        List of array max parameter of target sections

    See Also
    --------
    np.array

    Examples
    --------
    >>> l = get_arraymax_section_data(['INTEHEAD', 'LOGIHEAD'])
    >>> l
    array([1300000000, 1300000000])

    """
    result_list = []
    data_key_list = DATA_BLOCK_SPEC.keys()
    for key in key_list:
        if key in data_key_list:
            result_list.append(max_size)
        else:
            result_list.append(0)
    return np.array(result_list)


class PointerStruct:
    """Pointer struct.

        Parameters
        ----------
        size_inte :
        size_logi :
        size_doub :
        size_poro :
        size_permx :
        size_permy :
        size_permz :
    """
    def __init__(self, size_inte, size_logi, size_doub, size_poro, size_permx, size_permy, size_permz):
        ptr_dict = {}
        ptr_dict[INTEHEAD] = 0
        ptr_dict[LOGIHEAD] = ptr_dict[INTEHEAD] + size_inte
        ptr_dict[DOUBHEAD] = ptr_dict[LOGIHEAD] + size_logi
        ptr_dict[PORO] = ptr_dict[DOUBHEAD] + size_doub
        ptr_dict[PERMX] = ptr_dict[PORO] + size_poro
        ptr_dict[PERMY] = ptr_dict[PERMX] + size_permx
        ptr_dict[PERMZ] = ptr_dict[PERMY] + size_permy
        self.ptrs = ptr_dict
        self.size = size_inte + size_logi + size_doub + size_poro + size_permx + size_permy + size_permz


def save_init(field_rock, name, grid_dim,
              nactive, units_type, grid_type, sdate, i_phase, mode):
    """
       Save INIT target file

       Parameters
       ----------
       field_rock : np.array
       name : str
       grid_dim :

       Returns
       -------
    """
    if not (mode == 'w' or
            (mode == 'a' and
             (not os.path.isfile(name + '.INIT') or not os.path.isfile(name + '.INSPEC')))):
        return

    data = {}
    data['poro'] = field_rock.ravel(attr='PORO', inplace=False)
    data['permx'] = field_rock.ravel(attr='PERMX', inplace=False)
    data['permy'] = field_rock.ravel(attr='PERMY', inplace=False)
    data['permz'] = field_rock.ravel(attr='PERMZ', inplace=False)

    with open(name + '.INIT', 'w+b') as file_init:
        size_inte = write_unrst_section(file_init, INTEHEAD, META_BLOCK_SPEC[INTEHEAD], grid_dim,
                                        nactive, units_type, grid_type, sdate, i_phase)
        size_logi = write_unrst_section(file_init, LOGIHEAD, META_BLOCK_SPEC[LOGIHEAD])
        size_doub = write_unrst_section(file_init, DOUBHEAD, META_BLOCK_SPEC[DOUBHEAD])

        poro_items = data[PORO.lower()]
        INDEX_SECTIONS_DATA[PORO].number = poro_items.size
        size_poro = write_unrst_data_section_f(f=file_init, name=PORO, stype=DATA_BLOCK_SPEC[PORO].type,
                                               data_array=(poro_items))

        permx_items = data[PERMX.lower()]
        INDEX_SECTIONS_DATA[PERMX].number = permx_items.size
        size_permx = write_unrst_data_section_f(f=file_init, name=PERMX, stype=DATA_BLOCK_SPEC[PERMX].type,
                                                data_array=(permx_items))

        permy_items = data[PERMY.lower()]
        INDEX_SECTIONS_DATA[PERMY].number = permy_items.size
        size_permy = write_unrst_data_section_f(f=file_init, name=PERMY, stype=DATA_BLOCK_SPEC[PERMY].type,
                                                data_array=(permy_items))

        permz_items = data[PERMZ.lower()]
        INDEX_SECTIONS_DATA[PERMZ].number = permz_items.size
        size_permz = write_unrst_data_section_f(f=file_init, name=PERMZ, stype=DATA_BLOCK_SPEC[PERMZ].type,
                                                data_array=(permz_items))

        pointers = PointerStruct(size_inte, size_logi, size_doub, size_poro, size_permx, size_permy, size_permz)

    with open(name + '.INSPEC', 'w+b') as file_index:
        write_unrst_section(file_index, INTEHEAD, META_BLOCK_SPEC[INTEHEAD], grid_dim,
                            nactive, units_type, grid_type, sdate, i_phase)

        key_list = list(INDEX_SECTIONS_DATA.keys())
        write_unrst_data_section(f=file_index, name=NAME, stype=SPEC_DATA_BLOC_SPEC[NAME],
                                 data_array=np.array(list(map(format_keyword, key_list))))

        write_unrst_data_section(f=file_index, name=TYPE, stype=SPEC_DATA_BLOC_SPEC[TYPE],
                                 data_array=get_type_section_data(key_list))

        write_unrst_data_section_f(f=file_index, name=NUMBER, stype=SPEC_DATA_BLOC_SPEC[NUMBER],
                                   data_array=get_number_section_data(key_list))

        write_unrst_data_section_f(f=file_index, name=POINTER, stype=SPEC_DATA_BLOC_SPEC[POINTER],
                                   data_array=get_pointer_section_data(key_list, pointers))

        write_unrst_data_section_f(f=file_index, name=POINTERB, stype=SPEC_DATA_BLOC_SPEC[POINTERB],
                                   data_array=np.array([0] * len(key_list)))

        write_unrst_data_section_f(f=file_index, name=ARRAYMAX, stype=SPEC_DATA_BLOC_SPEC[ARRAYMAX],
                                   data_array=get_arraymax_section_data(key_list))

        write_unrst_data_section_f(f=file_index, name=ARRAYMIN, stype=SPEC_DATA_BLOC_SPEC[ARRAYMIN],
                                   data_array=np.array([0] * len(key_list)))
