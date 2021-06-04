"""Differentiable table interpolation."""
import torch
from torch import nn
from ...metamodelling.utils import LinearInterpolator, find_best_match_indices

EPS = 1e-8


class TableIOWrapper(nn.Module):
    """Wrapper allowing table-interpolation for tensors of arbitrary shapes."""
    def __init__(self, func):
        super().__init__()
        self._func = func

    def __call__(self, x, columns_dim=1, squeeze_output=True):
        """

        Parameters
        ----------
        x: torch.tensor
        columns_dim: int, None
        squeeze_output: bool

        Returns
        -------
        out: torch.tensor
        """
        scalar_input = False
        if x.ndim == 0:
            scalar_input = True
            x = x.unsqueeze(0)
        if columns_dim is None:
            shape = x.shape
            x = x.reshape(-1)
        else:
            x = x.transpose(columns_dim, 0)
            shape = x.shape[1:]
            x = x.reshape(x.shape[0], -1).transpose(0, 1)
            x = x.squeeze(1)
        n_points = x.shape[0]

        out = self._func(x)

        if out.numel() != n_points and out.ndim == 1:
            out = out.unsqueeze(0)
        if out.ndim == 1:
            out = out.view(*shape)
            if columns_dim is not None:
                out = out.unsqueeze(columns_dim)
        elif out.ndim == 2:
            out = out.transpose(1, 0)
            out = out.view(out.shape[0], *shape)
            if columns_dim is not None:
                out = out.transpose(0, columns_dim)
            if squeeze_output:
                out = out.squeeze(0 if columns_dim is None else columns_dim)
        if scalar_input:
            out = out.squeeze()
        return out


def get_callable_table(attr, table):
    """Returns callable interpolator for a table.

    Parameters
    ----------
    attr: str
        Table name.
    table: torch.Tensor

    Returns
    -------
    table_defined_function: callable
        table_defined_function(points) -> values
            points: torch.tensor
            values: torch.tensor
        First dimension of tensors should represent the number of points.
    """
    attr = attr.upper()
    if attr in TABLE_INTERPOLATOR:
        return TABLE_INTERPOLATOR[attr](table)
    return None


def _linear_table_interpolator(table, n_domain_columns=1, at_bounds='linear'):
    """Returns linear interpolation function for given table
    Parameters
    ----------
    table: torch.tensor
    n_domain_columns: int

    Returns
    -------
    table_defined_function: callable
        table_defined_function(points) -> values
            points: torch.tensor
            values: torch.tensor
        First dimension of tensors should represent the number of points.
    """
    if n_domain_columns != 1:
        raise NotImplementedError('Only 1-d linear table interpolation is supported.')
    points = table[:, :n_domain_columns].squeeze(1).clone()
    values = table[:, n_domain_columns:].squeeze(1).clone()
    return TableIOWrapper(
        LinearInterpolator(points, values, at_bounds=at_bounds)
    )


def _pvd_table_interpolator(table):
    """Returns inverse linear interpolation function for FVF and viscosity values
    Parameters
    ----------
    table: torch.tensor

    Returns
    -------
    table_defined_function: callable
        table_defined_function(points) -> values
            points: torch.tensor
            values: torch.tensor
        First dimension of tensors should represent the number of points.
    """
    inv_table = table.clone()
    inv_table[:, 1] = 1 / table[:, 1]
    inv_table[:, 2] = 1 / (table[:, 1] * table[:, 2])
    inv_linear_interpolator = _linear_table_interpolator(inv_table)

    @TableIOWrapper
    def interp(x):
        inv_y = inv_linear_interpolator(x.view(-1, 1))
        y = inv_y.clone()
        y[:, 1] = inv_y[:, 0] / inv_y[:, 1]
        y[:, 0] = 1 / inv_y[:, 0]
        y[y < EPS] = EPS
        return y

    return interp


def _relative_perm_table_interpolator(table):
    """Returns interpolation function for table with relative permeability curves
    Parameters
    ----------
    table: torch.tensor

    Returns
    -------
    table_defined_function: callable
        table_defined_function(points) -> values
            points: torch.tensor
            values: torch.tensor
        First dimension of tensors should represent the number of points.
    """
    return _linear_table_interpolator(table, at_bounds='constant')


def _split_pvto(table):
    rs_set = torch.unique(table[:, 0], sorted=True)
    branches = []
    saturated = []
    for rs in rs_set:
        mask = table[:, 0] == rs
        branch = table[mask]
        branches.append(branch[:, 1:])
        saturated.append(branch[0])
    saturated = torch.stack(saturated)
    return saturated, branches


def _pvto_table_interpolator(table):
    """Returns interpolation function for PVTO-table.
    Parameters
    ----------
    table: torch.tensor

    Returns
    -------
    table_defined_function: callable
        table_defined_function(points) -> values
            points: torch.tensor
            values: torch.tensor
        First dimension of tensors should represent the number of points.
    """
    pvto_sat, branches = _split_pvto(table)
    bubble_point_interpolator = _linear_table_interpolator(pvto_sat[:, :2])
    saturated_fvf_visc_interpolator = _pvd_table_interpolator(pvto_sat[:, 1:])
    branch_interpolators = [_pvd_table_interpolator(branch) for branch in branches]

    @TableIOWrapper
    def interp(x):
        p_bubble = bubble_point_interpolator(x[:, 0], columns_dim=None)
        fvf_visc_sat = saturated_fvf_visc_interpolator(p_bubble.unsqueeze(1))

        nearest_branch_indices = {
            'lower': find_best_match_indices(x[:, 0], pvto_sat[:, 0], less_or_equal=True),
            'upper': find_best_match_indices(x[:, 0], pvto_sat[:, 0], greater_or_equal=True)
        }
        nearest_branch_kinds = list(nearest_branch_indices.keys())
        for one, another in zip(nearest_branch_kinds, reversed(nearest_branch_kinds)):
            is_at_bound = nearest_branch_indices[one] == -1
            nearest_branch_indices[one][is_at_bound] = nearest_branch_indices[another][is_at_bound]

        fvf_visc_scaled = {kind: torch.empty_like(x) for kind in nearest_branch_kinds}
        nearest_rs = {kind: torch.empty_like(p_bubble) for kind in nearest_branch_kinds}

        for i, (branch, branch_interpolator) in enumerate(zip(branches, branch_interpolators)):
            for kind in nearest_branch_kinds:
                branch_mask = nearest_branch_indices[kind] == i
                if not branch_mask.any():
                    continue
                scaled_pressure = x[branch_mask, 1] - p_bubble[branch_mask] + branch[0, 0]
                fvf_visc = branch_interpolator(scaled_pressure.unsqueeze(1), columns_dim=1)

                fvf_visc_scaled[kind][branch_mask] = \
                    fvf_visc * fvf_visc_sat[branch_mask] / branch[0, 1:]
                nearest_rs[kind][branch_mask] = pvto_sat[i, 0]

        rs_delta = nearest_rs['upper'] - nearest_rs['lower']
        delta_mask = rs_delta != 0
        frac = torch.zeros_like(rs_delta)
        frac[delta_mask] = \
            (x[:, 0][delta_mask] - nearest_rs['lower'][delta_mask]) / rs_delta[delta_mask]
        frac = frac.unsqueeze(1)
        fvf_visc_output = fvf_visc_scaled['upper'] * frac + fvf_visc_scaled['lower'] * (1 - frac)
        fvf_visc_output[fvf_visc_output < EPS] = EPS
        return fvf_visc_output

    return interp


def _pvtw_table_interpolator(table):
    """Returns interpolation function for PVTW-table
    Parameters
    ----------
    table: _Table
    Returns
    -------
    table_defined_function
        table_defined_function(points) -> values
            points (pressure): torch.tensor
            values (FVF, VISC): torch.tensor
    """
    p, b, c, mu, viscosibility = table[0]

    @TableIOWrapper
    def interp(x):
        y = torch.stack(
            [b / (1 + c * (x - p) + c ** 2 * (x - p) ** 2 / 2),
             mu * torch.exp(viscosibility * (x - p))],
            dim=1
        )
        return y
    return interp


def get_baker_linear_model(swof, sgof, eps=1e-3):
    """Get Baker linear model.

    Parameters
    ----------
    swof: _Table
        SWOF table.
    sgof: _Table
        SGOF table.
    eps: float

    Returns
    -------
    table_defined_function
        table_defined_function(points) -> values
            points (SW, SG): torch.tensor
            values (KRO): torch.tensor
    """
    swof_interpolator = _relative_perm_table_interpolator(swof)
    sgof_interpolator = _relative_perm_table_interpolator(sgof)
    swc = swof[0, 0]

    @TableIOWrapper
    def interp(x):
        sw_plus_sg = x.sum(dim=1)
        sw_plus_sg_minus_swc = sw_plus_sg - swc
        _, krow, _ = swof_interpolator(sw_plus_sg, columns_dim=None)
        _, krog, _ = sgof_interpolator(sw_plus_sg_minus_swc, columns_dim=None)
        connected_water = x[:, 0] - swc < eps
        no_gas = x[:, 1] < eps
        other = torch.logical_not(no_gas + connected_water)

        sw_plus_sg_minus_swc = sw_plus_sg_minus_swc * other + torch.logical_not(other)
        kro_at_other = (x[:, 1] * krog + (x[:, 0] - swc) * krow) / sw_plus_sg_minus_swc

        kro = krog * connected_water + krow * no_gas + kro_at_other * other
        return kro
    return interp


TABLE_INTERPOLATOR = {
    None: _linear_table_interpolator,
    'PVDG': _pvd_table_interpolator,
    'PVDO': _pvd_table_interpolator,
    'PVTO': _pvto_table_interpolator,
    'PVTW': _pvtw_table_interpolator,
    'SWOF': _relative_perm_table_interpolator,
    'SGOF': _relative_perm_table_interpolator
}
