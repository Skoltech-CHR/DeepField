"""Methods for table interpolation"""
import numpy as np
import pandas as pd
from scipy.interpolate import LinearNDInterpolator, interp1d

def _linear_table_interpolator(table):
    """Returns linear interpolation function for given table
    Parameters
    ----------
    table: _Table

    Returns
    -------
    table_defined_function
        table_defined_function(points) -> values
            points: array-like of shape (n_points, input_dim)
            values: array-like of shape (n_points, output_dim)
    """
    domain_values = np.array(tuple(table.index.values))
    dependent_values = table.values
    if len(table.domain) < 2:
        return interp1d(domain_values, dependent_values, axis=0, bounds_error=False, fill_value='extrapolate')
    return LinearNDInterpolator(domain_values, dependent_values)

def _pvd_table_interpolator(table):
    """Returns inverse linear interpolation function for FVF and viscosity values
    Parameters
    ----------
    table: _Table

    Returns
    -------
    table_defined_function
        table_defined_function(points) -> values
            points: array-like of shape (n_points, input_dim)
            values: array-like of shape (n_points, output_dim)
    """
    def inv_interp(data):
        result = interp1d(domain_values, 1 / dependent_values, axis=0,
                          bounds_error=False, fill_value='extrapolate')(data)
        return np.vstack((1 / result[:, 0], result[:, 0] / result[:, 1])).T

    domain_values = table.index.values
    dependent_values = np.vstack((table.values[:, 0], table.values[:, 0] * table.values[:, 1])).T
    return inv_interp

def _relative_perm_table_interpolator(table):
    """Returns interpolation function for table with relative permeability curves
    Parameters
    ----------
    table: _Table

    Returns
    -------
    table_defined_function
        table_defined_function(points) -> values
            points: array-like of shape (n_points, input_dim)
            values: array-like of shape (n_points, output_dim)
    """
    domain_values = table.index.values
    dependent_values = table.values
    bound_vals = np.vstack((dependent_values[0], dependent_values[-1]))

    return interp1d(domain_values, dependent_values, axis=0, bounds_error=False,
                    fill_value=tuple(bound_vals))

def _pvto_table_interpolator(table):
    """Returns interpolation function for PVTO-table
    Parameters
    ----------
    table: _Table

    Returns
    -------
    table_defined_function
        table_defined_function(points) -> values
            points (pressure): array-like of shape (n_points, )
            values (FVF, VISC): array-like of shape (n_points, 2)
    """
    def pvto_update(undersat_branch, fvf_s, visc_s, fvf_u, visc_u):
        cols = undersat_branch.columns
        return pd.concat([undersat_branch[cols[0]]*fvf_s/fvf_u,
                          undersat_branch[cols[1]]*visc_s/visc_u], axis=1)

    def fvf_visc_o(data):
        df = pd.DataFrame(data, columns=['rs', 'press'])
        rspbub = pvto_sat.index.to_frame(index=False).set_index(pvto_sat.index.names[0])
        df['press_bub'] = interp1d(rspbub.index.values, rspbub.values, axis=0,
                                   bounds_error=False, fill_value='extrapolate')(df['rs'].values)
        ind_set_1 = np.where(np.isclose(df['press'], df['press_bub'], atol=1e-5))[0]
        df = df.assign(fvf=None, visc=None)
        df[['fvf', 'visc']] = _pvd_table_interpolator(pvto_sat.reset_index(level=0)[
            pvto_sat.columns])(df['press_bub'])
        ind_set_2 = list(set(np.arange(len(data))).difference(ind_set_1))
        for ind, row in zip(ind_set_2, df.iloc[ind_set_2].values):
            rs, p, pbub, fvf_s, visc_s = row
            undersat_vals = pd.concat([pvto_sat_2[pvto_sat_2.index.get_level_values(0) <= rs][-1:],
                                       pvto_sat_3[pvto_sat_3.index.get_level_values(0) >= rs][:1]])
            arr = []
            for ind2 in range(len(undersat_vals)):
                rs_u, pbub_u = undersat_vals.index[ind2]
                fvf_u, visc_u = undersat_vals.iloc[ind2].values
                pvto_current = pvto_update(pvto_undersat[np.isclose(pvto_undersat.index.get_level_values(0),
                                                                    rs_u)].copy(), fvf_s, visc_s, fvf_u, visc_u)
                arr.append(_pvd_table_interpolator(pvto_current.reset_index(level=0)[
                    pvto_current.columns])([p - pbub + pbub_u])[0])
            if len(arr) == 1:
                df.loc[ind, ['fvf', 'visc']] = arr[0]
            else:
                arr = np.array(arr)
                domain_values = undersat_vals.index.levels[0].values
                dependent_values = np.vstack((arr[:, 0], arr[:, 0] * arr[:, 1])).T
                res = interp1d(domain_values, 1 / dependent_values, axis=0,
                               bounds_error=False, fill_value='extrapolate')([rs])
                df.loc[ind, 'fvf'] = 1 / res[:, 0]
                df.loc[ind, 'visc'] = res[:, 0] / res[:, 1]
        return df[['fvf', 'visc']].values

    pvto_sat, pvto_undersat, pvto_sat_2, pvto_sat_3 = split_pvto(table)
    return fvf_visc_o

def _pvtw_table_interpolator(table):
    """Returns interpolation function for PVTW-table
    Parameters
    ----------
    table: _Table

    Returns
    -------
    table_defined_function
        table_defined_function(points) -> values
            points (pressure): array-like of shape (n_points, )
            values (FVF, VISC): array-like of shape (n_points, 2)
    """
    cols = table.columns

    def fvf_visc_w(press):
        p_ref = table.index.values.astype(np.float64)[0]
        fvf = table[cols[0]].values.astype(np.float64)[0]
        compr = table[cols[1]].values.astype(np.float64)[0]
        visc = table[cols[2]].values.astype(np.float64)[0]
        viscosibility = table[cols[3]].values.astype(np.float64)[0]
        fvf = fvf / (1 + compr * (press - p_ref) + compr**2 * (press - p_ref)**2 / 2)
        visc = visc * np.exp(viscosibility * (press - p_ref))
        return np.vstack((fvf, visc)).T
    return fvf_visc_w

def split_pvto(table):
    """Splits pvto-table for tables with saturated values and undersaturated branches
    Parameters
    ----------
    table: pandas.DataFrame
        PVTO-table.

    Returns
    -------
    pvto_sat : pandas.DataFrame
        Saturated part of PVTO-table.
    pvto_undersat : pandas.DataFrame
        PVTO-table with undersaturated branches.
    pvto_sat_2 : pandas.DataFrame
        Saturated part (last points of branches) of pvto_undersat-table.
    pvto_sat_3 : pandas.DataFrame
        Saturated part (first points of branches) of pvto_undersat-table.
    """
    index_names = table.index.names

    pvto_sat_index = table.index.to_frame(index=False).groupby(index_names[0], as_index=False).first()
    pvto_sat = table.loc[pd.MultiIndex.from_frame(pvto_sat_index)]
    pvto_counter = table.index.to_frame(index=False).groupby(index_names[0], as_index=False).count()
    pvto_undersat = table.loc[pvto_counter[pvto_counter[index_names[1]] > 1].set_index(index_names[0]).index]
    pvto_sat_2 = pvto_undersat.loc[pd.MultiIndex.from_frame(
        pvto_undersat.index.to_frame(index=False).groupby(index_names[0], as_index=False).last())] # last
    pvto_sat_3 = pvto_undersat.loc[pd.MultiIndex.from_frame(
        pvto_undersat.index.to_frame(index=False).groupby(index_names[0], as_index=False).first())]
    return pvto_sat, pvto_undersat, pvto_sat_2, pvto_sat_3

def baker_linear_model(tables, sat_w, sat_g, swc, eps=0.001):
    """Calculates relative permeability of oil by baker model
    Parameters
    ----------
    kr_ow: array-like
        Oil-water relative permeability.
    kr_og: array-like
        Oil-gas relative permeability.
    sat_w: array-like
        Water saturation.
    sat_g: array-like
        Gas saturation.
    swc: float
        Residual water saturation.
    eps: float
        Accuracy of comparison.

    Returns
    -------
    kr_o : array_like
        Relative permeability of oil.
    """
    _, kr_ow, _ = tables.swof(sat_g + sat_w).T
    _, kr_og, _ = tables.sgof(sat_g + sat_w - swc).T
    kr_o = np.zeros(sat_w.size)
    ind1 = np.where(sat_w - swc < eps)[0]
    kr_o[ind1] = kr_og[ind1]
    ind2 = np.where(sat_g < eps)[0]
    kr_o[ind2] = kr_ow[ind2]
    ind3 = list(set(np.arange(sat_w.size)).difference(set(ind1).union(ind2)))
    kr_o[ind3] = (sat_g[ind3]*kr_og[ind3] + (sat_w[ind3] - swc)*kr_ow[ind3])/(sat_g[ind3] + sat_w[ind3] - swc)
    return kr_o

TABLE_INTERPOLATOR = {None: _linear_table_interpolator,
                      'PVDG': _pvd_table_interpolator, 'PVDO': _pvd_table_interpolator,
                      'PVTO': _pvto_table_interpolator, 'PVTW': _pvtw_table_interpolator,
                      'SWOF': _relative_perm_table_interpolator, 'SGOF': _relative_perm_table_interpolator}
