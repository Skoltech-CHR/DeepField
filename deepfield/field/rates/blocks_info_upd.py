"""Methods to update information about grid blocks for a well segment."""
import warnings
import numpy as np
from ..grids import OrthogonalUniformGrid

def calculate_cf(rock, grid, segment, beta=1, units='METRIC', cf_aggregation='sum'):
    """Calculate connection factor values for each grid block of a segment."""
    if 'COMPDAT' in segment.attributes:
        return calculate_cf_compdat
    x_blocks, y_blocks, z_blocks = segment.blocks.T
    blocks_size = len(x_blocks)
    try:
        perm_tensor = np.vstack((rock.permx[x_blocks, y_blocks, z_blocks],
                                 rock.permy[x_blocks, y_blocks, z_blocks],
                                 rock.permz[x_blocks, y_blocks, z_blocks]))
    except AttributeError:
        return segment
    if units == 'METRIC':
        conversion_const = 0.00852702
    else:
        conversion_const = 0.00112712
    if isinstance(grid, OrthogonalUniformGrid):
        d_block = np.array([grid.cell_size]*blocks_size).T
    else:
        d_block = grid.cell_sizes((x_blocks, y_blocks, z_blocks)).T

    h_well = (segment.blocks_info[['Hx', 'Hy', 'Hz']].values.T *
              segment.blocks_info['PERF_RATIO'].values)
    r_well = segment.blocks_info['RAD'].values
    skin = segment.blocks_info['SKIN'].values
    wpi_mult = segment.blocks_info['MULT'].values
    if beta == 1:
        beta = np.array([beta]*blocks_size)
    else:
        beta = np.array(beta)
    d_1, d_2 = d_block[[1, 2, 0]], d_block[[2, 0, 1]]
    k_1, k_2 = perm_tensor[[1, 2, 0]], perm_tensor[[2, 0, 1]]
    k_h = (k_1 * k_2 * h_well**2)**0.5
    with warnings.catch_warnings(): # ignore devision by zero
        warnings.simplefilter("ignore")
        radius_equiv = (0.28 * (d_1**2 * np.sqrt(k_2 / k_1) + d_2**2 * np.sqrt(k_1 / k_2))**0.5 /
                        ((k_2 / k_1)**0.25 + (k_1 / k_2)**0.25))
        cf_projections = ((beta * wpi_mult * 2 * np.pi * conversion_const * k_h) /
                          (np.log(radius_equiv / r_well) + skin)).T
        if cf_aggregation == 'sum':
            segment.blocks_info['CF'] = cf_projections.sum(axis=1)
        elif cf_aggregation == 'eucl':
            segment.blocks_info['CF'] = np.sqrt((cf_projections ** 2).sum(axis=1))
        else:
            raise ValueError('Wrong value cf_aggregation={}, should be "sum" or "eucl".'.format(cf_aggregation))
    segment.blocks_info['CF'] = segment.blocks_info['CF'].fillna(0)
    return segment

def apply_perforations(segment, current_date=None):
    """Calculate perforation ratio for each grid block of the segment.

    ATTENTION: only latest perforation that covers the block
    defines the perforation ratio of this block.
    """
    if 'COMPDAT' in segment.attributes:
        return apply_perforations_compdat(segment, current_date)

    if current_date is None:
        perf = segment.perf
    else:
        perf = segment.perf.loc[segment.perf['DATE'] < current_date]

    col_rad = 'DIAM' if 'DIAM' in perf.columns else 'RAD'

    b_info = segment.blocks_info
    b_info['PERF_RATIO'] = 0

    blocks_start_md = b_info['MD'].values
    last_block_size = np.linalg.norm(b_info.tail(1)[['Hx', 'Hy', 'Hz']])
    blocks_end_md = np.hstack([blocks_start_md[1:],
                               [blocks_start_md[-1] + last_block_size]])
    blocks_size = blocks_end_md - blocks_start_md

    for line in perf[['MDL', 'MDU', col_rad, 'SKIN', 'MULT', 'CLOSE']].values:
        md_start, md_end, rad, skin, wpimult, close_flag = line
        is_covered = (blocks_end_md > md_start) & (blocks_start_md <= md_end)
        if not is_covered.any():
            continue
        b_info.loc[is_covered, 'SKIN'] = skin
        b_info.loc[is_covered, 'MULT'] = wpimult
        b_info.loc[is_covered, 'RAD'] = rad/2 if 'DIAM' in perf.columns else rad
        covered_ids = np.where(is_covered)[0]
        if close_flag:
            b_info.loc[covered_ids, 'PERF_RATIO'] = 0
            continue
        first = covered_ids.min()
        last = covered_ids.max()
        #full perforation of intermediate blocks
        b_info.loc[first+1:last, 'PERF_RATIO'] = 1
        if first == last:
            #partial perforation of the single block
            perf_size = md_end - md_start
            b_info.loc[first, 'PERF_RATIO'] = min(perf_size / blocks_size[first], 1)
            continue
        #partial perforation of the first block
        perf_size = blocks_end_md[first] - md_start
        b_info.loc[first, 'PERF_RATIO'] = min(perf_size / blocks_size[first], 1)
        #partial perforation of the last block
        perf_size = md_end - blocks_start_md[last]
        b_info.loc[last, 'PERF_RATIO'] = min(perf_size / blocks_size[last], 1)
    return segment

def apply_perforations_compdat(segment, current_date=None):
    """Update `blocks_info` table with perforations parameters from `COMPDAT`."""
    if current_date is None:
        compdat = segment.compdat
    else:
        compdat = segment.compdat.loc[segment.compdat['DATE'] < current_date]

    cf = np.full(segment.blocks.shape[0], np.nan)
    skin = np.full(segment.blocks.shape[0], np.nan)
    perf_ratio = np.zeros(segment.blocks.shape[0])

    for i, line in compdat.iterrows():
        condition = np.where(
            np.logical_and(
                np.logical_and(segment.blocks[:, 0] == line['I'] - 1,
                               segment.blocks[:, 1] == line['J'] - 1),
                np.logical_and(segment.blocks[:, 2] >= line['K1'] - 1,
                               segment.blocks[:, 2] <= line['K2'] - 1)))
        if line['MODE'] == 'SHUT':
            close_flag = True
        elif line['MODE'] == 'OPEN':
            close_flag = False
        else:
            raise ValueError(('Incorrect mode `{}` in line {} of COMPDAT ' +
                              'for well `{}`').format(
                                  line['MODE'], i, segment.name))
        perf_ratio[condition] = 0 if close_flag else 1
        skin[condition] = line['SKIN'] if 'SKIN' in line else 0
        cf[condition] = line['CF'] if 'CF' in line else np.nan

    segment.blocks_info['PERF_RATIO'] = perf_ratio
    if 'CF' in segment.compdat.columns:
        segment.blocks_info['CF'] = cf
    segment.blocks_info['SKIN'] = skin

    return segment

def calculate_cf_compdat(segment):
    "Calculate connection factors from `COMPDAT` table."
    if 'CF' not in segment.blocks_info.columns:
        raise Exception(('No `CF` in `blocks_info` table of well {}. ' +
                         'Probably no `CF` in `COMPDAT`').format(segment.name))
    return segment
