"""Plot utils."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # pylint: disable=unused-import
from mpl_toolkits.mplot3d import Axes3D # pylint: disable=unused-import
from ipywidgets import interact, widgets
import pyvista as pv

COLORS = ['r', 'b', 'm', 'g']


def show_cube_static(data, x=None, y=None, z=None, t=None, figsize=None, aspect='auto', **kwargs):
    """Plot a given cube slice.

    Parameters
    ----------
    data : ndarray
        A 3D or 4D array to get slices from.
    x : int or None
        Slice along x-axis to show.
    y : int or None
        Slice along y-axis to show.
    z : int or None
        Slice along z-axis to show.
    t : int or None
        Slice along t-axis to show.
    figsize : array-like, optional
        Output plot size.
    aspect : str
        Aspect ratio. Default is 'auto'.
    kwargs : dict
        Additional keyword arguments for plot.

    Returns
    -------
    Plot of a cube slice.
    """
    count = np.sum([i is not None for i in [x, y, z, t]])
    if data.ndim == 4:
        if count != 2:
            raise ValueError('Two slices are expected for spatio-temporal data, found {}.'.format(count))
    elif data.ndim == 3:
        if count != 1:
            raise ValueError('Single slice is expected for spatial data, found {}.'.format(count))
    else:
        raise ValueError('Data should have 3 or 4 dimensions, found {}.'.format(data.ndim))

    if data.ndim == 3:
        data = np.expand_dims(data, 0)
        t = 0
    slices = tuple(slice(i) if i is None else i for i in [t, x, y, z])

    plt.figure(figsize=figsize)
    plt.imshow(data[slices].T, aspect=aspect, **kwargs)
    plt.show()


def show_cube_interactive(data, figsize=None, aspect='auto', **kwargs):
    """Plot 3 cube slices with interactive sliders.

    Parameters
    ----------
    data : ndarray
        A 3D array to show.
    figsize : array-like, optional
        Output plot size.
    aspect : str
        Aspect ratio. Default is 'auto'.
    kwargs : dict
        Additional keyword arguments for plot.

    Returns
    -------
    Plot of 3 cube slices with interactive sliders.
    """
    def update(t=None, x=0, y=0, z=0):
        data3d = data if t is None else data[t]

        _, axes = plt.subplots(1, 3, figsize=figsize)
        axes[0].imshow(data3d[x].T, aspect=aspect, **kwargs)
        axes[0].axvline(y, c='r')
        axes[0].axhline(z, c='r')

        axes[1].imshow(data3d[:, y, :].T, aspect=aspect, **kwargs)
        axes[1].axvline(x, c='r')
        axes[1].axhline(z, c='r')

        axes[2].imshow(data3d[:, :, z], aspect=aspect, **kwargs)
        axes[2].axvline(y, c='r')
        axes[2].axhline(x, c='r')

        axes[0].set_title('X')
        axes[1].set_title('Y')
        axes[2].set_title('Z')

    shape = data.shape

    if data.ndim == 3:
        interact(lambda x, y, z: update(None, x, y, z),
                 x=widgets.IntSlider(value=shape[0] / 2, min=0, max=shape[0] - 1, step=1),
                 y=widgets.IntSlider(value=shape[1] / 2, min=0, max=shape[1] - 1, step=1),
                 z=widgets.IntSlider(value=shape[2] / 2, min=0, max=shape[2] - 1, step=1))
    elif data.ndim == 4:
        interact(update,
                 t=widgets.IntSlider(value=shape[0] / 2, min=0, max=shape[0] - 1, step=1),
                 x=widgets.IntSlider(value=shape[1] / 2, min=0, max=shape[1] - 1, step=1),
                 y=widgets.IntSlider(value=shape[2] / 2, min=0, max=shape[2] - 1, step=1),
                 z=widgets.IntSlider(value=shape[3] / 2, min=0, max=shape[3] - 1, step=1))
    else:
        raise ValueError('Invalid data shape. Expected 3 or 4, got {}.'.format(data.ndim))
    plt.show()


def plot_bounds_3d(top, bottom, figsize=None):
    """Plot top and bottom surfaces."""
    x = np.arange(top.shape[0])
    y = np.arange(top.shape[1])
    x, y = np.meshgrid(x, y)
    nans = np.isnan(top)

    plt.figure(figsize=figsize)
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, bottom.T, color='b', edgecolor='none')
    ax.plot_surface(x, y, top.T, color='r', edgecolor='none')
    ax.set_zlim(bottom[~nans].max(), top[~nans].min())
    plt.show()


def plot_bounds_2d(top, bottom, x=None, y=None, figsize=None):
    """Plot top and bottom lines in x and y projections."""
    nans = np.isnan(top)
    ylim = (bottom[~nans].max(), top[~nans].min())

    def update(x=0, y=0):
        _, axes = plt.subplots(1, 2, figsize=figsize)
        axes[0].plot(top[x], label='top')
        axes[0].plot(bottom[x], label='bottom')
        axes[0].axvline(y, c='r')
        axes[0].set_xlim((0, top.shape[1]))
        axes[0].set_title('x')

        axes[1].plot(top[:, y], label='top')
        axes[1].plot(bottom[:, y], label='bottom')
        axes[1].axvline(x, c='r')
        axes[1].set_xlim((0, top.shape[0]))
        axes[1].set_title('y')

        for ax in axes:
            ax.legend()
            ax.set_ylim(*ylim)

    if (x is None) and (y is None):
        interact(update,
                 x=widgets.IntSlider(0, 0, top.shape[0] - 1, step=1),
                 y=widgets.IntSlider(0, 0, top.shape[1] - 1, step=1))
    else:
        x = 0 if x is None else x
        y = 0 if y is None else y
        update(x, y)
    plt.show()

def make_patch_spines_invisible(ax):
    """Make patch spines invisible"""
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_table_1d(table, figsize=None):
    """
    Plot table with 1-dimensional domain
    Parameters
    ----------
    table: geology.src.tables.tables._Table
        Table to be plotted
    figsize: tuple
    """
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    fig.suptitle(table.name)
    ax.set_xlabel(table.domain[0])
    ax = [ax, ]
    ax_position = 0.8
    for _ in range(len(table.columns) - 1):
        ax_position += 0.2
        ax.append(ax[0].twinx())
        ax[-1].spines["right"].set_position(("axes", ax_position))
        make_patch_spines_invisible(ax[-1])
        ax[-1].spines["right"].set_visible(True)

    x = table.index.values
    for i, col in enumerate(table.columns):
        ax[i].plot(x, table[col].values, color=COLORS[i])
        ax[i].set_ylabel(col, color=COLORS[i])
        ax[i].tick_params(axis='y', labelcolor=COLORS[i])
    plt.show()


def plot_table_2d(table, figsize=None):
    """
    Plot table with 2-dimensional domain
    Parameters
    ----------
    table: geology.src.tables.tables._Table
        Table to be plotted
    figsize: tuple
    """
    domain_names = list(table.domain)
    domain0_value_widget = widgets.SelectionSlider(
        description=domain_names[0],
        options=list(sorted(set(table.index.get_level_values(0))))
    )

    def update(domain0_value):
        cropped_table = table.loc[table.index.get_level_values(0) == domain0_value]
        cropped_table = cropped_table.droplevel(0)
        cropped_table.domain = [domain_names[1]]
        plot_table_1d(cropped_table, figsize)
    interact(update, domain0_value=domain0_value_widget)


def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int)
    poly.lines = cells
    return poly
