"""Parse utils for ECLIPSE binary files."""
import math
from struct import unpack
import numpy as np

DATA_TYPES = {
    'INTE': (4, 'i', 1000),
    'REAL': (4, 'f', 1000),
    'LOGI': (4, 'i', 1000),
    'DOUB': (8, 'd', 1000),
    'CHAR': (8, '8s', 105),
    'MESS': (8, '8s', 105),
    'C008': (8, '8s', 105)
}


def _get_type_info(data_type):
    """Returns element size, format and element skip for the given data type.

    Parameters
    ----------
    data_type: str
        Should be a key from the DATA_TYPES

    Returns
    -------
    type_info: tuple
    """
    try:
        return DATA_TYPES[data_type]
    except KeyError as exc:
        raise ValueError('Unknown datatype %s.' % data_type) from exc


def read_ecl_bin(path, attrs=None, decode=True, sequential=False, subset=None, logger=None):
    """Reads binary ECLIPSE file into a dictionary.

    Parameters
    ----------
    path : str
        Path to the binary file to be read
    attrs : list or None, optional
        List of keys to be read (e.g. ['SOIL', 'SWAT']).
        If None is given all keys will be read from the file.
    decode : bool
        Return decoded data. Default True.
    sequential : bool
        If True, returns a list of partiotions for each attributes. Makes scence e.g. for states data.
        Default False.
    subset : array of integers, optional
        If given, returns a subset of partitions for sequential data.
    logger : logger, optional
        Event logger.

    Returns
    -------
    sections : dict
        Data from the binary file.
    """
    if attrs is not None:
        attrs = [attr.strip().upper() for attr in attrs]
    _, sections = _read_sections(path, attrs, decode=decode, sequential=sequential,
                                 subset=subset, logger=logger)
    return sections

def _read_sections(path, attrs=None, decode=True, sequential=False, subset=None, logger=None):
    """Reads binary ECLIPSE file into a dictionary.

    Parameters
    ----------
    path : str
        Path to the binary file to be read
    attrs : list or None, optional
        List of keys to be read (e.g. ['SOIL', 'ROCK']).
        If None is given all keys will be read from the file.
    decode : bool
        Return decoded data. Default True.
    sequential : bool
        If True, returns a list of partiotions for each attributes. Makes scence e.g. for states data.
        Default False.
    subset : array of integers, optional
        If given, returns a subset of partitions for sequential data.
    logger : logger, optional
        Event logger.

    Returns
    -------
    data: dict
        Data from the binary file in the form of dict.
        data[key] contains BINARY data and data type info
    """
    def logger_print(msg, level='info'):
        if logger is not None:
            getattr(logger, level)(msg)

    def is_in_subset(i):
        if subset is None:
            return True
        return i in subset

    if subset is not None and not sequential:
        raise ValueError('Subset can be specified for sequentional data only.')

    sections_counter = {} if attrs is None else {attr: 0 for attr in attrs}

    logger_print("Start reading {}".format(path))
    with open(path, 'rb') as f:
        header = f.read(4)
        sections = dict()
        while True:
            try:
                section_name = unpack('8s', f.read(8))[0].decode('ascii').strip().upper()
            except:  # pylint: disable=bare-except
                break
            n_elements = unpack('>i', f.read(4))[0]
            data_type = unpack('4s', f.read(4))[0].decode('ascii')
            f.read(8)
            element_size, fmt, element_skip = _get_type_info(data_type)
            f.seek(f.tell() - 24)
            binary_data = f.read(24 + element_size * n_elements + 8 * (math.floor((n_elements - 1) / element_skip) + 1))
            if (attrs is None) or (section_name in attrs):
                sections_counter[section_name] = sections_counter.get(section_name, 0) + 1
                if sequential and section_name not in sections:
                    sections[section_name] = []
                if not sequential and section_name in sections:
                    raise ValueError('Got multiple partitions for non-sequentional attribute {}. '\
                                     'Try to set sequential=True.'.format(section_name))
                section = (n_elements, data_type, element_size, fmt, element_skip, binary_data)
                i = sections_counter[section_name] - 1
                if is_in_subset(i):
                    if decode:
                        if sequential:
                            logger_print("Decoding {} at timestep {}.".format(section_name, i))
                        else:
                            logger_print("Decoding {}.".format(section_name))
                        section = _decode_section(section)
                    if sequential:
                        sections[section_name].append(section)
                    else:
                        sections[section_name] = section
    if attrs is not None:
        for attr in attrs:
            if attr not in sections.keys():
                logger_print('{} was not found in file.'.format(attr), level='warning')
    logger_print("Finish reading {}".format(path))
    return header, sections

def _decode_section(section):
    """Decodes section of a binary ECLIPSE file.

    Parameters
    ----------
    section: list

    Returns
    -------
    decoded_section: ndarray
    """
    n_elements, data_type, element_size, fmt, element_skip, binary_data = section

    n_skip = math.floor((n_elements - 1) / element_skip)
    skip_elements = 8 // element_size
    skip_elements_total = n_skip * skip_elements
    data_format = fmt * (n_elements + skip_elements_total)
    data_size = element_size * (n_elements + skip_elements_total)
    if data_type in ['INTE', 'REAL', 'LOGI', 'DOUB']:
        data_format = '>' + data_format
    decoded_section = list(unpack(data_format, binary_data[24: 24 + data_size]))
    del_ind = np.repeat(np.arange(1, 1 + n_skip) * element_skip, skip_elements)
    del_ind += np.arange(len(del_ind))
    decoded_section = np.delete(decoded_section, del_ind)
    if data_type in ['CHAR', 'C008']:
        decoded_section = np.char.decode(decoded_section, encoding='ascii')
    return decoded_section
