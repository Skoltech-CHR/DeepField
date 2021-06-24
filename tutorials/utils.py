"""Helper functions for tutorials."""
from math import ceil
import numpy as np
import matplotlib.pyplot as plt
import torch
from deepfield.datasets.transforms import AddBatchDimension, RemoveBatchDimension
from deepfield.metamodelling.utils import get_model_device


def make_loss_plot(train_loss):
    """Plot loss."""
    train_loss_legend = np.array(train_loss).T
    plt.figure()
    plt.title('Training curve')
    plt.ylabel('Loss')
    plt.xlabel('#iter')
    plt.semilogy(train_loss_legend[0], train_loss_legend[1], 'b-', label='Train')
    plt.legend()
    plt.show()


def make_crossplots(true_states, predicted_states, attrs, truth_name, model_name, well_mask=None):
    """Make crossplots."""
    if not isinstance(true_states, np.ndarray):
        true_states = true_states.detach().numpy()
    if not isinstance(predicted_states, np.ndarray):
        predicted_states = predicted_states.detach().numpy()
    if true_states.ndim == 5:
        true_states = true_states.reshape(true_states.shape[:2] + (-1, ))
    if predicted_states.ndim == 5:
        predicted_states = predicted_states.reshape(predicted_states.shape[:2] + (-1, ))
    points = np.random.choice(true_states.shape[2], size=true_states.shape[2] // 50)
    true_states = true_states[:, :, points]
    predicted_states = predicted_states[:, :, points]
    if well_mask is not None:
        if not isinstance(well_mask, np.ndarray):
            well_mask = well_mask.detach().numpy()
        if well_mask.ndim == 3:
            well_mask = well_mask.reshape(-1)
        well_mask = well_mask[points]
        s_wm = true_states[..., well_mask == 1]
        s_pred_wm = predicted_states[..., well_mask == 1]
    else:
        s_wm, s_pred_wm = true_states, predicted_states
    fig, ax = plt.subplots(nrows=ceil(len(attrs) / 3), ncols=3, figsize=(12, 3 * ceil(len(attrs) / 3)))
    fig.suptitle('Crossplots')
    for i, ch in enumerate(attrs):
        row, col = i // 3, i % 3
        maxima = np.max([true_states[:, i], predicted_states[:, i]])
        minima = np.min([true_states[:, i], predicted_states[:, i]])
        ax[row, col].axis([minima, maxima, minima, maxima])
        ax[row, col].plot([minima, maxima], [minima, maxima], 'k--')
        ax[row, col].plot(true_states[:, i].ravel(), predicted_states[:, i].ravel(), 'b.', alpha=0.01)
        if well_mask is not None:
            ax[row, col].plot(s_wm[:, i].ravel(), s_pred_wm[:, i].ravel(), '.', color='orange', alpha=0.1)
        if row == 1:
            ax[row, col].set_xlabel(truth_name)
        if col == 0:
            ax[row, col].set_ylabel(model_name)
        ax[row, col].set_title(ch)
    plt.show()


def make_comparison_plots(true_states, predicted_states, attrs, truth_name, model_name, well_mask=None,
                          plot_quantiles=False):
    """Make comparison plots: prediction versus time."""
    if not isinstance(true_states, np.ndarray):
        true_states = true_states.detach().numpy()
    if not isinstance(predicted_states, np.ndarray):
        predicted_states = predicted_states.detach().numpy()
    if true_states.ndim == 5:
        true_states = true_states.reshape(true_states.shape[:2] + (-1, ))
    if predicted_states.ndim == 5:
        predicted_states = predicted_states.reshape(predicted_states.shape[:2] + (-1, ))
    if well_mask is not None:
        if not isinstance(well_mask, np.ndarray):
            well_mask = well_mask.detach().numpy()
        if well_mask.ndim == 3:
            well_mask = well_mask.reshape(-1)
        s_wm = true_states[..., well_mask == 1]
        s_pred_wm = predicted_states[..., well_mask == 1]
        s_wm_mean, s_pred_wm_mean = s_wm.mean(axis=2), s_pred_wm.mean(axis=2)
    else:
        s_wm_mean, s_pred_wm_mean = true_states.mean(axis=2), predicted_states.mean(axis=2)
    s_np_mean = true_states.mean(axis=2)
    s_pred_np_mean = predicted_states.mean(axis=2)
    if plot_quantiles:
        s_np_05, s_np_95 = np.percentile(true_states, 5, axis=2), np.percentile(true_states, 95, axis=2)
        s_pred_np_05 = np.percentile(predicted_states, 5, axis=2)
        s_pred_np_95 = np.percentile(predicted_states, 95, axis=2)
    time = np.arange(true_states.shape[0])
    fig, ax = plt.subplots(nrows=ceil(len(attrs) / 3), ncols=3, figsize=(12, 3 * ceil(len(attrs) / 3)))
    fig.suptitle('Comparison')
    for i, ch in enumerate(attrs):
        row, col = i // 3, i % 3
        ax[row, col].set_title(ch)
        if plot_quantiles:
            ax[row, col].fill_between(time, s_np_05[:, i], s_np_95[:, i], color='b', alpha=0.1)
            ax[row, col].fill_between(time, s_pred_np_05[:, i], s_pred_np_95[:, i], color='m', alpha=0.1)
        ax[row, col].plot(time, s_np_mean[:, i], 'b-', label=truth_name)
        ax[row, col].plot(time, s_pred_np_mean[:, i], 'm-', label=model_name)
        if well_mask is not None:
            ax[row, col].plot(time, s_wm_mean[:, i], 'b--')
            ax[row, col].plot(time, s_pred_wm_mean[:, i], 'm--')
        if col == 0 and row == 0:
            ax[row, col].legend()
    plt.show()


def predict(sample, lsd, kwargs, ae_kwargs, max_seq_len, verbose=True):
    """Make metamodel prediction."""
    if not sample.state.batch_dimension:
        sample.transformed(AddBatchDimension, inplace=True)
        batch_dimension = False
    else:
        batch_dimension = True

    pieces = np.arange(0, sample.masks.time.shape[-1] + max_seq_len, max_seq_len)
    states = []
    latent = []
    states_ae = []
    latent_ae = []
    state_init = sample.states[:, 0]
    model_device = get_model_device(lsd)
    sample_device = sample.device

    with torch.no_grad():
        latent_init = lsd.state_enc(state_init.to(model_device), **ae_kwargs)
        rock_reduced = lsd.params_enc(sample.rock.to(model_device), **ae_kwargs)
        for left, right in zip(pieces[:-1], pieces[1:]):
            t_i = sample.masks.time[:, left:right]
            control_i, control_t_i = lsd.get_control_subset(
                sample.control, sample.masks.control_t, (t_i[0, 0], t_i[0, -1])
            )
            control_reduced = lsd.control_enc(control_i.to(model_device), **ae_kwargs)
            latent_piece = lsd._compute_dynamics(  # pylint: disable=protected-access
                (latent_init, rock_reduced, control_reduced),
                control_t_i.to(model_device), t_i.to(model_device), **kwargs
            )
            states_piece = lsd._decode(latent_piece, **ae_kwargs)  # pylint: disable=protected-access
            latent_ae_piece = lsd.state_enc(sample.states[:, left:right].to(model_device), **ae_kwargs)
            states_ae_piece = lsd._decode(latent_ae_piece, **ae_kwargs)  # pylint: disable=protected-access
            latent.append(latent_piece.to(sample_device))
            states.append(states_piece.to(sample_device))
            latent_ae.append(latent_ae_piece.to(sample_device))
            states_ae.append(states_ae_piece.to(sample_device))
            latent_init = latent_piece[:, -1]

    states = torch.cat(states, dim=1)
    latent = torch.cat(latent, dim=1)
    states_ae = torch.cat(states_ae, dim=1)
    latent_ae = torch.cat(latent_ae, dim=1)

    if verbose:
        actnum = sample.masks.actnum
        latent_loss = (latent - latent_ae).pow(2).mean().item()
        ae_loss = (sample.states - states_ae).transpose(0, -actnum.ndim)[..., actnum == 1].pow(2).mean().item()
        mean_loss = (sample.states - states).transpose(0, -actnum.ndim)[..., actnum == 1].pow(2).mean().item()
        print('Latent loss:\t%f' % latent_loss)
        print('Autoencoder loss: \t%f' % ae_loss)
        print('Variational inference loss: \t%f' % (latent_loss + ae_loss))
        print('Mean loss (active cells): %f' % mean_loss)
        for i, ch in enumerate(sample.sample_attrs.states):
            channel_loss = (sample.states[:, :, i] - states[:, :, i])
            channel_loss = channel_loss.transpose(0, -actnum.ndim)[..., actnum == 1].pow(2).mean().item()
            print('\t%s loss: \t%f' % (ch, channel_loss))

    old_states = sample.states
    delattr(sample, 'states')
    sample_predicted = sample.copy()
    sample_predicted.states = states
    sample_autoencoder = sample.copy()
    sample_autoencoder.states = states_ae
    sample.states = old_states

    if not batch_dimension:
        sample.transformed(RemoveBatchDimension, inplace=True)
        sample_predicted.transformed(RemoveBatchDimension, inplace=True)
        sample_autoencoder.transformed(RemoveBatchDimension, inplace=True)
    return sample_predicted, sample_autoencoder
