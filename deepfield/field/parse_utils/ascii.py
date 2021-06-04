"""Parser utils."""
import os
import re
from io import StringIO
from pathlib import Path
from itertools import zip_longest
import numpy as np
import pandas as pd
import chardet

_COLUMN_LENGTH = 13

IGNORE_SECTIONS = ['ARITHMETIC', 'COPY', 'MULTIPLY',
                   'RPTISOL', 'RPTPROPS', 'RPTREGS',
                   'RPTRUNSP', 'RPTSCHED', 'RPTSMRY', 'RPTSOL']

DEFAULT_ENCODINGS = ['utf-8', 'cp1251']

class StringIteratorIO:
    """String iterator for text files."""
    def __init__(self, path, encoding=None):
        self._path = path
        if (encoding is not None) and encoding.startswith('auto'):
            encoding = encoding.split(':')
            if len(encoding) > 1:
                n_bytes = int(encoding[1])
            else:
                n_bytes = 5000
            with open(self._path, 'rb') as file:
                raw = file.read(n_bytes)
                self._encoding = chardet.detect(raw)['encoding']
        else:
            self._encoding = encoding
        self._line_number = 0
        self._f = None
        self._buffer = ''
        self._last_line = None
        self._on_last = False
        self._proposed_encodings = DEFAULT_ENCODINGS.copy()

    @property
    def line_number(self):
        """Number of lines read."""
        return self._line_number

    def __iter__(self):
        return self

    def __next__(self):
        if self._on_last:
            self._on_last = False
            return self._last_line
        try:
            line = next(self._f).split('--')[0]
        except UnicodeDecodeError:
            return self._better_decoding()
        self._line_number += 1
        if line.strip():
            self._last_line = line
            return line
        return next(self)

    def _better_decoding(self):
        """Last chance to read line with default encodings."""
        try:
            enc = self._proposed_encodings.pop()
        except IndexError as err:
            raise UnicodeDecodeError('Failed to decode at line {}'.format(self._line_number + 1)) from err
        if enc == self._encoding:
            return self._better_decoding()
        self._f = open(self._path, 'r', encoding=enc) #pylint: disable=consider-using-with
        self._encoding = enc
        for _ in range(self._line_number):
            next(self._f)
        return next(self)

    def prev(self):
        """Set current position to previous line."""
        if self._on_last:
            raise ValueError("Maximum cache depth is reached.")
        self._on_last = True
        return self

    def __enter__(self):
        self._f = open(self._path, 'r', encoding=self._encoding) #pylint: disable=consider-using-with
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _ = exc_type, exc_val, exc_tb
        self._f.close()

    def read(self, n=None):
        """Read n characters."""
        while not self._buffer:
            try:
                self._buffer = next(self)
            except StopIteration:
                break
        result = self._buffer[:n]
        self._buffer = self._buffer[len(result):]
        return result

    def skip_to(self, stop, *args):
        """Skip strings until stop token."""
        if isinstance(stop, str):
            stop = [stop]
        stop_pattern = '|'.join([x + '$' for x in stop])
        for line in self:
            if re.match(stop_pattern, line.strip(), *args):
                return

def preprocess_path(path):
    """Parse a string with path to Path instance."""
    parts = path.split('.')
    try:
        path_str = '.'.join(parts[:-1] + [parts[-1].rstrip(' /').split()[0]])
    except IndexError:
        return Path(path)
    path_str = path_str.strip(' \t\n\'"').replace('\\', '/')
    return Path(path_str)

def _insensitive_match(where, what):
    """Find unique 'what' in directory 'where' ignoring 'what' case."""
    found = [p for p in os.listdir(where) if p.lower() == what.lower()]
    if len(found) > 1:
        raise FileNotFoundError("Multiple paths found for {} in {}".format(what, where))
    if len(found) == 0:
        raise FileNotFoundError("Path {} does not exists in {}".format(what, where))
    return found[0]

def case_insensitive_path(path):
    """Resolve system path given path in arbitrary case."""
    parts = path.parts
    result = Path(parts[0])
    for p in parts[1:]:
        result = result / Path(_insensitive_match(str(result), p))
    return result

def _get_path(line, data_dir, logger, raise_errors):
    """Case insensitive file path parser."""
    path = preprocess_path(line)
    look_path = (data_dir / path).resolve()
    try:
        actual_path = case_insensitive_path(look_path)
    except FileNotFoundError as err:
        if raise_errors:
            raise FileNotFoundError(err) from None
        logger.warning("Ignore missing file {}.".format(str(look_path)))
        return None
    return actual_path

def tnav_ascii_parser(path, loaders_map, logger, data_dir=None, encoding=None, raise_errors=False):  # pylint: disable=too-many-branches, too-many-statements
    """Read tNav ASCII files and call loaders."""
    data_dir = path.parent if data_dir is None else data_dir
    filename = path.name
    logger.info("Start reading {}".format(filename))
    with StringIteratorIO(path, encoding=encoding) as lines:
        for line in lines:
            firstword = line.split(maxsplit=1)[0].upper()
            if firstword in IGNORE_SECTIONS:
                lines.skip_to('/')
            elif firstword in ['EFOR', 'EFORM', 'HFOR', 'HFORM']:
                column_names = line.split()[1:]
            elif firstword == 'ETAB':
                if 'ETAB' in loaders_map:
                    logger.info("[{}:{}] Loading ETAB".format(filename, lines.line_number))
                    loaders_map['ETAB'](lines, column_names=column_names)
                else:
                    lines.skip_to(['/', 'ENDE'])
            elif firstword == 'TTAB':
                if 'TTAB' in loaders_map:
                    logger.info("[{}:{}] Loading TTAB".format(filename, lines.line_number))
                    loaders_map['TTAB'](lines)
                else:
                    lines.skip_to('ENDT')
            elif (firstword in ['EFIL', 'EFILE', 'TFIL']) and (firstword in loaders_map):
                line = next(lines)
                include = _get_path(line, data_dir, logger, raise_errors)
                if include is None:
                    continue
                with StringIteratorIO(include, encoding=encoding) as inc_lines:
                    logger.info("[{0}:{1}] Loading {2} from {3}"\
                                    .format(filename, lines.line_number, firstword, include))
                    if firstword == 'TFIL':
                        loaders_map[firstword](inc_lines)
                    else:
                        loaders_map[firstword](inc_lines, column_names=column_names)
            elif firstword == 'HTAB' and firstword in loaders_map:
                logger.info("[{}:{}] Loading HTAB".format(filename, lines.line_number))
                loaders_map['HTAB'](lines, column_names=column_names)
            elif (firstword in ['HFIL', 'HFILE']) and (firstword in loaders_map):
                line = next(lines)
                include = _get_path(line, data_dir, logger, raise_errors)
                if include is None:
                    continue
                with StringIteratorIO(include, encoding=encoding) as inc_lines:
                    logger.info("[{0}:{1}] Loading {2} from {3}"\
                                    .format(filename, lines.line_number, firstword, include))
                    loaders_map[firstword](inc_lines, column_names=column_names)
            elif firstword in ['INCLUDE', 'USERFILE']:
                line = next(lines)
                include = _get_path(line, data_dir, logger, raise_errors)
                if include is None:
                    continue
                logger.info("[{0}:{1}] Include {2}".format(filename, lines.line_number, include))
                tnav_ascii_parser(include, loaders_map, logger, data_dir=data_dir,
                                  encoding=encoding, raise_errors=raise_errors)
            elif (firstword == 'WELLTRACK') and (firstword in loaders_map):
                lines.prev() #pylint: disable=not-callable
                logger.info("[{0}:{1}] Loading {2}".format(filename, lines.line_number, firstword))
                loaders_map[firstword](lines)
            elif (firstword in ['GROU', 'GROUP']) and (firstword in loaders_map):
                lines.prev() #pylint: disable=not-callable
                logger.info("[{0}:{1}] Loading {2}".format(filename, lines.line_number, firstword))
                loaders_map[firstword](lines)
            elif (firstword in ['AQCO', 'AQCT']) and (firstword in loaders_map):
                lines.prev() #pylint: disable=not-callable
                logger.info("[{0}:{1}] Loading {2}".format(filename, lines.line_number, firstword))
                loaders_map[firstword](lines)
            elif (firstword in ['AQUANCON', 'AQUCT']) and (firstword in loaders_map):
                logger.info("[{0}:{1}] Loading {2}".format(filename, lines.line_number, firstword))
                loaders_map[firstword](lines)
            elif firstword in loaders_map:
                logger.info("[{0}:{1}] Loading {2}".format(filename, lines.line_number, firstword))
                loaders_map[firstword](lines)
    logger.info("Finish reading {}".format(filename))

def decompress_array(s, dtype=None):
    """Extracts compressed numerical array from ASCII string.
    Interprets A*B as B repeated A times."""
    if dtype is None:
        dtype = float
    nums = []
    for x in s.split():
        try:
            val = [dtype(float(x))]
        except ValueError:
            k, val = x.split('*')
            val = [dtype(val)] * int(k)
        nums.extend(val)
    return np.array(nums)

def read_array(buffer, dtype=None, compressed=True, **kwargs):
    """Read array data from a string buffer before first occurrence of '/' symbol.

    Parameters
    ----------
    buffer : buffer
        String buffer to read.
    dtype : dtype or None
        Defines dtype of an output array. If not specified, float array is returned.
    compressed : bool
        If True, A*B will be interpreted as B repeated A times.

    Returns
    -------
    arr : ndarray
        Parsed array.
    """
    _ = kwargs
    arr = []
    last_line = False
    if dtype is None:
        dtype = float
    for line in buffer:
        if '/' in line:
            last_line = True
            line = line.split('/')[0]
        if compressed:
            x = decompress_array(line, dtype=dtype)
        else:
            x = np.fromstring(line.strip(), dtype=dtype, sep=' ')
        if x.size:
            arr.append(x)
        if last_line:
            break
    return np.hstack(arr)

def read_table(buffer, table_info, dtype=None):
    """Read numerical table data from a string buffer before first occurrence of non-digit line.

    Parameters
    ----------
    buffer: buffer
        String buffer to read.
    table_info: dict
        Dict with table's meta information:
            table_info['attrs'] - list of column names
            table_info['domain'] - list of domain columns indices
    dtype: dtype or None
        Defines dtype of an output array. If not specified, float array is returned.

    Returns
    -------
    table : pandas DataFrame
        Parsed table.
    """
    table = []
    group = []
    group_sep = '/'
    is_group_end = False
    n_columns = None
    n_attrs = len(table_info['attrs'])

    if dtype is None:
        dtype = float
    for line in buffer:
        line = line.strip()
        if not line[0].isdigit():
            if group:
                group = np.stack(group, axis=0)
                group[np.isnan(group)] = group[0, 0]
                table.append(group)
            table = np.concatenate(table, axis=0)
            buffer.prev()
            break
        if group_sep in line:
            line = line.split(group_sep)[0] #re.sub(group_sep, '', line)
            is_group_end = True

        x = np.fromstring(line, dtype=dtype, sep=' ')

        if x.size:
            n_columns = x.shape[0] if n_columns is None else n_columns
            fixed_n_of_columns = n_columns == x.shape[0]

            if n_attrs != x.shape[0] and not fixed_n_of_columns:
                x = np.concatenate([np.full(n_attrs - x.shape[0], np.nan), x])
            if n_attrs != x.shape[0]:
                if table_info['defaults'] is None or table_info['defaults'][x.shape[0]] is None:
                    raise ValueError('Default values are not assumed for this position in the table.')
                x = np.concatenate([x, np.array(table_info['defaults'][x.shape[0]:])])

            group.append(x)
        if is_group_end:
            if group:
                group = np.stack(group, axis=0)
                group[np.isnan(group)] = group[0, 0]
                table.append(group)
            is_group_end = False
            group = []

        n_columns = x.shape[0]

    table = pd.DataFrame(table, columns=table_info['attrs'])

    if table_info['domain'] is not None:
        domain_attrs = np.array(table_info['attrs'])[table_info['domain']]
    else:
        domain_attrs = np.array([])
    if domain_attrs.shape[0] == 1:
        table = table.set_index(domain_attrs[0])
    elif domain_attrs.shape[0] > 1:
        multi_index = pd.MultiIndex.from_frame(table[domain_attrs])
        table = table.drop(domain_attrs, axis=1)
        table = table.set_index(multi_index)
    return table

def read_rsm(filename, logger):
    """Parse *.RSM files to dict."""
    result = {}
    blocks = _rsm_blocks(filename)
    for block in blocks:
        block_res = _parse_block(block, logger)
        result = _update_result(result, block_res, logger)
    return result

def _rsm_blocks(filename):
    """TBD."""
    block = None
    block_start_re = re.compile(r'\d+\n')
    with open(filename) as f:
        line = f.readline()
        while line:
            if block_start_re.fullmatch(line):
                if block is not None:
                    yield ''.join(block)
                block = []
            if block is not None:
                block.append(line)
            line = f.readline()
        if block is not None:
            yield ''.join(block)

def _split_block(block):
    """TBD."""
    lines = block.split('\n')
    border_re = re.compile(r'\s\-+')
    border_count = 0

    i = None

    for i, line in enumerate(lines):
        if border_re.fullmatch(line):
            border_count += 1
        if border_count == 3:
            break

    if border_count == 3:
        return '\n'.join(lines[:i+1]), '\n'.join(lines[i+1:])
    raise ValueError('Block can not be splitted')

def _parse_header(header):
    """TBD."""
    header_data = header.split('\n')[4:-1]
    names = _split_string(header_data[0])
    units = _split_string(header_data[1])
    has_multiplyers = len(header_data[2].strip()) > 0 and header_data[2].strip()[0] == '*'
    multiplyers = (_split_string(header_data[2]) if has_multiplyers
                   else _split_string(''.join('\t' * len(header_data[2]))))
    multiplyers = [_parse_rsm_multiplyer(mult) for mult in multiplyers]
    obj_string_number = 3 if has_multiplyers else 2
    objects = _split_string(header_data[obj_string_number])
    numbers = (_split_string(header_data[obj_string_number+1]) if
               len(header_data) > (obj_string_number + 1) else [''] * len(objects))

    return names, units, multiplyers, objects, numbers

def _split_string(string, n_sym=_COLUMN_LENGTH):
    """TBD."""
    return [''.join(s).strip() for s  in grouper(string, n_sym)]

def _parse_rsm_multiplyer(multiplyer):
    """Parse rsm multiplyer value"""
    if multiplyer == '':
        return 1
    match = re.fullmatch(r'\*(\d+)\*\*(\d+)', multiplyer)
    if match is not None:
        return int(match.groups()[0]) ** int(match.groups()[1])
    raise ValueError('Wrong `multiplyer` format')

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def _parse_data(data):
    """TBD."""
    return np.loadtxt(StringIO(data))

def _parse_block(block, logger):
    """TBD."""
    header, data = _split_block(block)
    names, units, multiplyers, objects, numbers = _parse_header(header)
    num_data = _parse_data(data)

    res = {}

    for i, (
            obj, name, unit, multiplyer, number
        ) in enumerate(zip(objects, names, units, multiplyers, numbers)):
        if obj == '':
            obj = '_global'

        if obj not in res:
            res[obj] = {}
        current_obj = res[obj]
        if number != '':
            if '_children' not in res[obj]:
                res[obj]['_children'] = {}
            if number not in res[obj]['_children']:
                res[obj]['_children'][number] = {}
            current_obj = res[obj]['_children'][number]

        if name not in current_obj:
            current_obj[name] = {
                'units': unit,
                'multiplyer': multiplyer,
                'data': num_data[:, i].reshape(-1)
            }

        else:
            logger.warn(('Object {} already contains field {}. \
                          New data is ignored.').format(obj, name))

    return res

def _update_result(result, new_data, logger):
    """TBD."""
    for obj, obj_data in new_data.items():
        if obj not in result:
            result[obj] = obj_data
        else:
            if isinstance(result[obj], dict):
                result[obj] = _update_result(result[obj], obj_data, logger)
            else:
                is_equal = ((obj_data == result[obj]).all()
                            if isinstance(obj_data, np.ndarray)
                            else obj_data == result[obj])
                if not is_equal:
                    logger.warn(('New value of {} {} is not equal to old value {}. \
                                 Data was rewrited.').format(obj, obj_data, result[obj]))
                    result[obj] = obj_data
    return result

def parse_perf_line(line, column_names, defaults):
    """Get wellname and perforation from a single event file line.
    Expected format starts with: WELL 'DD.MM.YYYY' PERF/PERFORATION
    """
    vals = line.split()
    well = vals[0].strip("'\"")
    date = pd.to_datetime(vals[1], format='%d.%m.%Y', errors='coerce')
    vals = [v.upper() for v in vals]
    if 'BRANCH' in vals[3:]:
        well = ":".join([well, vals[vals.index('BRANCH') + 1]])
    data = {'WELL': well, 'DATE': date}
    for i, v in enumerate(vals[3:len(column_names) + 1]):
        if '*' in v:
            k = int(v.split('*', 1)[0])
            for j in range(k):
                name = column_names[2 + i + j]
                data[name] = [defaults[name]]
        else:
            name = column_names[2 + i]
            data[name] = [float(v)]
    data['CLOSE'] = 'CLOSE' in vals[3:]
    return pd.DataFrame(data)

def parse_control_line(line, mode_control, value_control):
    """Get wellname and control from a single event file line.
    Expected format starts with: WELL 'DD.MM.YYYY'
    """
    vals = line.split()
    data = {'WELL' : [vals[0].strip("'\"")],
            'DATE': [pd.to_datetime(vals[1], format='%d.%m.%Y', errors='coerce')]}
    vals = [v.upper() for v in vals]
    mode = [k for k in vals[2:] if k in mode_control]
    if mode:
        if len(mode) > 1:
            raise ValueError("Multiple mode controls.")
        data['MODE'] = [mode[0]]
    for i, k in enumerate(vals[2:]):
        if k in value_control:
            try:
                data[k] = [float(vals[i + 3])]
            except ValueError:
                data[k] = None
    return pd.DataFrame(data)

def parse_history_line(line, column_names):
    """Get data from a single history file line.
    Expected format starts with: WELL 'DD.MM.YYYY'
    """
    vals = line.split()
    well = vals[0].strip("'\"")
    date = pd.to_datetime(vals[1], format='%d.%m.%Y', errors='coerce')
    vals = [v.upper() for v in vals]
    data = {'WELL': well, 'DATE': date}
    for i, name in enumerate(column_names[2:]):
        data[name] = [float(vals[2 + i])]
    return pd.DataFrame(data)

def read_dates_from_buffer(buffer, attr, logger):
    """Read keywords representing output dates (ARRAY, DATES).

    Parameters
    ----------
    buffer: buffer
        String buffer to read from.
    attr: str
        Keyword to read.
    logger: Logger
        Log info handler.

    Returns
    -------
    output_dates: list
    """
    _ = attr
    buffer = buffer.prev()
    args = next(buffer).strip('\n').split()[1:]

    if args[0] != 'DATE':
        logger.warning('ARRAy of type {} is not supported and is ignored.'.format(args[0]))
        return None

    dates = []
    for line in buffer:
        if '/' in line:
            break
        dates.append(line)
    return pd.to_datetime(dates)

def dates_to_str(dates):
    """Transforms list of dates into a string representation with the ARRAY keyword heading.

    Parameters
    ----------
    dates: list

    Returns
    -------
    dates: str
    """
    heading = 'ARRAY DATE'
    footing = '/'
    res = [heading] + [d.strftime('%d %b %Y').upper() for d in dates] + [footing]
    res = '\n'.join(res) + '\n\n'
    return res
