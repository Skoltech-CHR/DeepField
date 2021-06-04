"""Grid utils."""
import itertools

import numpy as np
from numba import njit


@njit
def add_det(array_of_a, results):
    """Add determinat of matrices stacked in input array to result array."""
    for i in range(len(array_of_a)):  # pylint: disable=consider-using-enumerate
        results[i] += abs(np.linalg.det(array_of_a[i]))


@njit
def numba_get_volumes(xyz_corn):
    """Compute cell volumes."""
    volumes = np.zeros(xyz_corn.shape[:3]).ravel()
    v1 = xyz_corn[:, :, :, 1] - xyz_corn[:, :, :, 0]
    v2 = xyz_corn[:, :, :, 2] - xyz_corn[:, :, :, 0]
    v3 = xyz_corn[:, :, :, 4] - xyz_corn[:, :, :, 0]
    a = np.stack((v1, v2, v3), axis=-1).reshape((-1, 3, 3))
    add_det(a, volumes)

    v1 = xyz_corn[:, :, :, 1] - xyz_corn[:, :, :, 5]
    v2 = xyz_corn[:, :, :, 4] - xyz_corn[:, :, :, 5]
    v3 = xyz_corn[:, :, :, 6] - xyz_corn[:, :, :, 5]
    a = np.stack((v1, v2, v3), axis=-1).reshape((-1, 3, 3))
    add_det(a, volumes)

    v1 = xyz_corn[:, :, :, 2] - xyz_corn[:, :, :, 1]
    v2 = xyz_corn[:, :, :, 4] - xyz_corn[:, :, :, 1]
    v3 = xyz_corn[:, :, :, 6] - xyz_corn[:, :, :, 1]
    a = np.stack((v1, v2, v3), axis=-1).reshape((-1, 3, 3))
    add_det(a, volumes)

    v1 = xyz_corn[:, :, :, 1] - xyz_corn[:, :, :, 3]
    v2 = xyz_corn[:, :, :, 2] - xyz_corn[:, :, :, 3]
    v3 = xyz_corn[:, :, :, 7] - xyz_corn[:, :, :, 3]
    a = np.stack((v1, v2, v3), axis=-1).reshape((-1, 3, 3))
    add_det(a, volumes)

    v1 = xyz_corn[:, :, :, 1] - xyz_corn[:, :, :, 5]
    v2 = xyz_corn[:, :, :, 7] - xyz_corn[:, :, :, 5]
    v3 = xyz_corn[:, :, :, 6] - xyz_corn[:, :, :, 5]
    a = np.stack((v1, v2, v3), axis=-1).reshape((-1, 3, 3))
    add_det(a, volumes)

    v1 = xyz_corn[:, :, :, 2] - xyz_corn[:, :, :, 1]
    v2 = xyz_corn[:, :, :, 7] - xyz_corn[:, :, :, 1]
    v3 = xyz_corn[:, :, :, 6] - xyz_corn[:, :, :, 1]
    a = np.stack((v1, v2, v3), axis=-1).reshape((-1, 3, 3))
    add_det(a, volumes)
    return volumes.reshape(xyz_corn.shape[:3]) / 6


@njit
def isclose(a, b, rtol=1e-05, atol=1e-08):
    """np.isclose."""
    return abs(a - b) <= (atol + rtol * abs(b))


@njit
def numba_get_xyz(dimens, zcorn, coord):
    """Get x, y, z coordinates of cell vertices."""
    nx, ny, _ = dimens
    xyz = np.zeros(zcorn.shape[:3] + (8, 3))
    xyz[..., 2] = zcorn
    shifts = [(0, 0), (-1, 0), (0, -1), (-1, -1)]
    for i in range(nx + 1):
        for j in range(ny + 1):
            line = coord[i, j]
            top_point = line[:3]
            vec = line[3:] - line[:3]
            if isclose(vec[2], 0):
                if not isclose(vec[0], 0) & isclose(vec[1], 0):
                    vec[2] = 1e-10
                else:
                    is_degenerated = True
            else:
                is_degenerated = False
            for k in range(8):
                ik = i + shifts[k % 4][0]
                jk = j + shifts[k % 4][1]
                if (ik < 0) or (ik >= nx) or (jk < 0) or (jk >= ny):
                    continue
                if is_degenerated:
                    xyz[ik, jk, :, k] = top_point
                else:
                    z_along_line = (zcorn[ik, jk, :, k] - top_point[2]) / vec[2]
                    xyz[ik, jk, :, k, :2] = top_point[:2] + vec[:2] * z_along_line.reshape((-1, 1))
    return xyz


def get_top_z_coords(zcorn, actnum):
    """For each coordinate line get z coordinate of the less deepest active cell.
    If coordinate line contains no active cells, the result is NaN."""
    if actnum.dtype is not np.dtype(bool):
        actnum = actnum.astype(bool)
    nx, ny = actnum.shape[:2]
    z_top = np.zeros((nx + 1, ny + 1))
    global_depth = zcorn.max()  # z increases with depth

    top_z_indices = np.argmax(actnum, axis=-1)
    x, y = np.indices((nx, ny))
    top_faces = zcorn[x, y, top_z_indices, 0]
    top_faces_act_mask = actnum[x, y, top_z_indices]
    top_faces[~top_faces_act_mask] = global_depth

    z_arrs = np.stack([top_faces[1:, 1:, 0],
                       top_faces[:-1, 1:, 1],
                       top_faces[1:, :-1, 2],
                       top_faces[:-1, :-1, 3]], axis=-1)
    z_top[1:-1, 1:-1] = z_arrs.min(axis=-1)

    z_arrs = np.stack([top_faces[0, :-1, 2], top_faces[0, 1:, 0]], axis=-1)
    z_top[0, 1:-1] = z_arrs.min(axis=-1)

    z_arrs = np.stack([top_faces[-1, :-1, 3], top_faces[-1, 1:, 1]], axis=-1)
    z_top[-1, 1:-1] = z_arrs.min(axis=-1)

    z_arrs = np.stack([top_faces[:-1, 0, 1], top_faces[1:, 0, 0]], axis=-1)
    z_top[1:-1, 0] = z_arrs.min(axis=-1)

    z_arrs = np.stack([top_faces[:-1, -1, 3], top_faces[1:, -1, 2]], axis=-1)
    z_top[1:-1, -1] = z_arrs.min(axis=-1)

    z_top[0, 0] = top_faces[0, 0, 0]
    z_top[0, -1] = top_faces[0, -1, 2]
    z_top[-1, 0] = top_faces[-1, 0, 1]
    z_top[-1, -1] = top_faces[-1, -1, 3]

    act_lines = np.zeros_like(z_top, dtype=bool)
    act_lines[:-1, :-1][top_faces_act_mask] = True
    act_lines[1:, :-1][top_faces_act_mask] = True
    act_lines[:-1, 1:][top_faces_act_mask] = True
    act_lines[1:, 1:][top_faces_act_mask] = True

    z_top[~act_lines] = np.nan
    return z_top


def get_connectivity_matrix(actnum, connectivity):
    """Get connectivity matrix.

    Parameters
    ----------
    actnum : numpy.ndarray
        Active cells mask.
    connectivity : int
        Connectivity index.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray)
        Tuple consisted of connectivity matrix and invalid cells mask.
    """
    dimens = actnum.shape
    active_indices = np.where(actnum)
    res = []
    invalid_cells = []
    for d in itertools.product(*[(-1, 0, 1)] * 3):
        n = np.sum(np.abs(d))
        if 0 < n <= connectivity:
            tmp = np.stack(active_indices)
            invalid_cells_tmp = np.zeros(tmp.shape[1:], bool)
            for i, z in enumerate(d):
                tmp[i] += z

            for i in range(3):
                invalid_cells_tmp[tmp[i] > (dimens[i] - 1)] = True
                tmp[:, tmp[i] > (dimens[i] - 1)] = 0

            invalid_cells_tmp[(tmp < 0).any(axis=0)] = True
            tmp[:, (tmp < 0).any(axis=0)] = 0
            invalid_cells_tmp[np.logical_not(actnum[tmp[0], tmp[1], tmp[2]])] = True
            tmp[:, np.logical_not(actnum[tmp[0], tmp[1], tmp[2]])] = 0
            res.append(tmp)
            invalid_cells.append(invalid_cells_tmp)

    res = np.stack(res)
    invalid_cells = np.stack(invalid_cells)
    invalid_cells = np.moveaxis(invalid_cells, (0, 1), (1, 0))
    res = np.moveaxis(res, (0, 1, 2), (1, 2, 0))
    return res, invalid_cells
