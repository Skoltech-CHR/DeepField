"""Notebook utils."""
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian

def loss_plot(loss, logscale=False, save=None):
    """Loss plot."""
    fig, ax = plt.subplots(1, figsize=(10, 5))
    fig.patch.set_facecolor('xkcd:white')
    if logscale:
        plt.semilogy(loss, lw=3)
    else:
        plt.plot(loss, lw=3)
    plt.xlabel('Iterations', fontsize=18)
    plt.ylabel('Loss function value', fontsize=18)
    ax.tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    plt.show()

def hist_plots(rock, connect, save=None):
    """Historgams of the correction factors."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('xkcd:white')

    ax[0].hist(rock, bins=30, range=(-1, 1), density=True)
    ax[0].axvline(rock.mean(), c='r', lw=3)
    ax[0].set_title('Rock corrections', fontsize=22)
    ax[0].tick_params(axis='both', labelsize=18)

    ax[1].hist(connect, bins=30, range=(-1, 3), density=True)
    ax[1].set_title('Connectivity corrections', fontsize=22)
    ax[1].tick_params(axis='both', labelsize=18)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    plt.show()

def slice_view(initial_params, final_params, well_mask, z_ind,
               name='values', cv=1.5, save=None):
    """Plot z-slice of 3D arrays."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 6))
    fig.patch.set_facecolor('xkcd:white')

    # initial properties
    ax[0].imshow(gaussian(initial_params[:, :, z_ind], 1),
                 cmap='bwr', vmax=cv, vmin=-cv)
    ax[0].set_title('Initial {}'.format(name), fontsize=22)
    ax[0].set_ylabel('Y', fontsize=18)

    # final properties
    ax[1].imshow(gaussian(final_params[:, :, z_ind], 1),
                 cmap='bwr', vmax=cv, vmin=-cv)
    ax[1].set_title('Final {}'.format(name), fontsize=22)

    # difference
    diff = final_params[:, :, z_ind] - initial_params[:, :, z_ind]
    im = ax[2].imshow(gaussian(diff, 1), cmap='bwr', vmax=cv, vmin=-cv)
    ax[2].set_title('Difference', fontsize=22)

    # wells position
    for i in range(3):
        ax[i].scatter(*np.where(well_mask[:, :, z_ind].T), c='black', s=60)
        ax[i].set_xlabel('X', fontsize=18)
        ax[i].tick_params(axis='x', labelsize=18)
        ax[i].tick_params(axis='y', labelsize=18)

    fig.subplots_adjust(bottom=0.21)
    cbar_ax = fig.add_axes([0.278, 0.05, 0.5, 0.03])
    cbar_ax.tick_params(axis='x', labelsize=18)
    fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    if save is not None:
        plt.savefig(save)
    plt.show()

def cumulative_plots(target_rates, pred_rates, phases, vline=None, save=None):
    """Cumulative rates plot."""
    target_cum = np.cumsum(target_rates, axis=0)
    pred_cum = np.cumsum(pred_rates, axis=0)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('xkcd:white')
    for i, k in enumerate(phases):
        ax[i].plot(target_cum[:, i], label='Target', lw=4)
        ax[i].plot(pred_cum[:, i], label='HM', lw=4)
        ax[i].set_title(k, fontsize=22)
        ax[i].legend(fontsize=18)
        ax[i].set_xlabel('Days', fontsize=18)
        if vline is not None:
            ax[i].axvline(vline, c='gray')
        ax[i].tick_params(axis='x', labelsize=18)
        ax[i].tick_params(axis='y', labelsize=18)
        ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax[0].set_ylabel('Cumulative volume', fontsize=18)
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    plt.show()

def corr_plots(target_rates, pred_rates, phases, mark_well=None, save=None):
    """Scatterplot target vs predicted."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('xkcd:white')
    colors = ['b'] * len(target_rates)
    if mark_well is not None:
        colors[mark_well] = 'r'
    for i, k in enumerate(phases):
        x = pred_rates[..., i].sum(axis=1)
        y = target_rates[..., i].sum(axis=1)
        ax[i].scatter(x, y, c=colors, s=90)
        ax[i].set_xlabel('Cumulative predicted', fontsize=18)
        ax[i].plot([0, x.max()], [0, x.max()], c='k')
        corr = np.corrcoef(x, y)[0, 1]
        ax[i].set_title(k + ', R={:.2f}'.format(corr), fontsize=22)
        ax[i].ticklabel_format(style='sci', axis='both', scilimits=(0, 0))
        ax[i].tick_params(axis='x', labelsize=18)
        ax[i].tick_params(axis='y', labelsize=18)
        ax[i].set_aspect('equal')
    ax[0].set_ylabel('Cumulative target', fontsize=18)
    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    plt.show()

def gas_oil_plot(gas_targ, gas_pred, oil_targ, oil_pred, vline=None, save=None):
    """Gas/oil ratio plot."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    fig.patch.set_facecolor('xkcd:white')

    ax.plot(gas_targ/oil_targ, label='Target', lw=4)
    ax.plot(gas_pred/oil_pred, label='HM', lw=4)

    ax.set_title('Gas/oil ratio', fontsize=22)
    ax.legend(fontsize=18)
    ax.set_xlabel('Days', fontsize=18)
    if vline is not None:
        ax.axvline(vline, c='gray', lw=2)
    ax.tick_params(axis='x', labelsize=18)
    ax.tick_params(axis='y', labelsize=18)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.tight_layout()
    if save is not None:
        plt.savefig(save)
    plt.show()
