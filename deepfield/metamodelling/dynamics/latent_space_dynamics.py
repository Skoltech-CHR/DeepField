"""Latent space dynamics."""
#pylint: disable=wrong-import-order,wrong-import-position
import sys
import torch

sys.path.insert(0, '..')
from ...field.utils import get_control_interval_mask

from .._base_nn_model import BaseModel
from ..losses import standardize_loss_pattern
from ..utils import get_model_device


class LatentSpaceDynamics(BaseModel):
    """Module for modelling dynamics in a learnable latent space."""
    _attrs = ('states', 'rock', 'control', ('masks', 'control_t'), ('masks', 'time'))
    _ae_kwargs = ('use_checkpointing', 'max_in_batch')

    def __init__(self, state_enc, state_dec, params_enc, control_enc, dynamics):
        """
        Parameters
        ----------
        state_enc: nn.Module
            Encoder module for states.
        state_dec: nn.Module
            Decoder module for states.
        params_enc: nn.Module
            Encoder module for rock (other parameters will be added later).
        control_enc: nn.Module
            Encoder module for control (transformed to the spatial form).
        dynamics: nn.Module
            Module, which computes the derivative of states over time, given params and instantaneous control.
        """
        super().__init__()
        self.state_enc = state_enc
        self.state_dec = state_dec
        self.params_enc = params_enc
        self.control_enc = control_enc
        self.dynamics = dynamics

    def forward(self, init_state, params, control, control_t, t, *args, get_latent_states=False, **kwargs):
        """
        Parameters
        ----------
        init_state: torch.Tensor
            Initial state. Shape: (B, C_s, *dims)
        params: torch.Tensor
            Params tensor (rock mostly). Shape: (B, C_p, *dims)
        control: torch.Tensor
            Spatial control tensor. Shape: (B, T_c, C_c, *dims)
        control_t: torch.Tensor
            Tensor holding a sequence of time points for which control is given.
            Each time must be strictly larger than the previous time. Shape: (B, T_c)
        t: torch.Tensor
            Tensor holding a sequence of time points for which to solve for states.
            The initial time point (i.e. time point of the init_state) should be the first
            element of this sequence, and each time must be strictly larger than the previous time.
            Shape: (B, T)
        args: tuple
            Any additional args passed to the encoders, decoder and dynamics modules.
        get_latent_states: bool
            If True, return solutions in the latent space as a second argument.
        kwargs: dict

        Returns
        -------
        solution: torch.Tensor
            Shape: (B, T, C_s, *dims)
        latent_solution: torch.Tensor
            Returned only if `get_latent_states=True`.
            Shape: (B, T, C_l, *latent_dims)
        """
        ae_kwargs = self._parse_ae_kwargs(kwargs)

        assert torch.all(t - t[:1] == 0), "Solution timesteps should be equal for batch items."
        control, control_t = self.get_control_subset(control, control_t, (t[0, 0], t[0, -1]))
        device = next(self.parameters()).device
        control = control.to(device)

        encoded = self._encode(init_state, params, control, *args, **ae_kwargs)
        r_states = self._compute_dynamics(encoded, control_t, t, *args, **kwargs)
        decoded = self._decode(r_states, *args, **ae_kwargs)

        return (decoded, r_states) if get_latent_states else decoded

    @staticmethod
    def get_control_subset(control, control_t, time_interval):
        """Returns a control subset that corresponds to the target result time interval."""
        assert torch.all(control_t - control_t[:1] == 0), "Control timesteps should be equal for batch items."
        mask = get_control_interval_mask(control_t[0], time_interval)
        if not torch.any(mask):
            raise ValueError('No control found for time interval ({}, {}).'
                             .format(time_interval[0], time_interval[1]))
        return control[:, mask], control_t[:, mask]

    def _encode(self, init_state, params, control, *args, **kwargs):
        """Encode initial state, params and control."""
        r_init_state = self.state_enc(init_state, *args, **kwargs)
        r_params = self.params_enc(params, *args, **kwargs)
        r_control = self.control_enc(control, *args, **kwargs)
        return r_init_state, r_params, r_control

    def _compute_dynamics(self, encoded, control_t, t, *args, **kwargs):
        """Compute latent space dynamics."""
        r_states = self.dynamics(*encoded, control_t, t, *args, **kwargs)
        return r_states

    def _decode(self, r_states, *args, **kwargs):
        """Decode latent states."""
        return self.state_dec(r_states, *args, **kwargs)

    def _parse_ae_kwargs(self, kwargs):
        """Filter kwargs which are not suitable for autoencoding modules."""
        ae_kwargs = {}
        for key in kwargs:
            if key in self._ae_kwargs:
                ae_kwargs[key] = kwargs[key]
        return ae_kwargs

    def make_training_iter(self, sample, loss_pattern, latent_loss_pattern=(), autoencoder_loss_pattern=(),
                           loss_between_differences=False, tbptt_step=None, tbptt_tail=None,
                           tbptt_loss_between_steps=True, **kwargs):

        if (tbptt_step is not None and tbptt_tail is None) or (tbptt_step is None and tbptt_tail is not None):
            raise ValueError('If TBPTT is assumed to be used, both `k1` and `k2` should be specified.')
        if tbptt_step is None and tbptt_tail is None:
            return self._make_training_iter(
                sample, loss_pattern, latent_loss_pattern, autoencoder_loss_pattern, loss_between_differences, **kwargs
            )
        return self._make_tbptt_training_iter(
            sample, loss_pattern, latent_loss_pattern, autoencoder_loss_pattern, loss_between_differences,
            tbptt_step, tbptt_tail, tbptt_loss_between_steps, **kwargs
        )

    def _make_training_iter(self, sample, loss_pattern, latent_loss_pattern, autoencoder_loss_pattern,
                            loss_between_differences, **kwargs):
        """Make one training iteration. Runs backward pass. Return loss of the prediction.

        Parameters
        ----------
        sample: dict
            Sample from a dataset. Should contain 'states', 'control', 'rock', 'time' and all the required masks.
        loss_pattern: tuple
            Loss pattern between ground truth and predicted states.
        latent_loss_pattern: tuple
            Loss pattern between encoded ground truth states and predicted by dynamics latent space states.
        autoencoder_loss_pattern: tuple
            Loss pattern between ground truth states and encoded-decoded states.
        loss_between_differences: bool
            If True, losses are computed not between states, but rather between "gradients" (differences) of states.
        kwargs: dict
            Any additional named arguments passed to forward methods of composing modules.

        Returns
        -------
        loss
            Detached loss.
        """
        inp = self._get_attrs_from_sample(sample, *self._attrs)

        latent_loss_pattern = standardize_loss_pattern(latent_loss_pattern)
        autoencoder_loss_pattern = standardize_loss_pattern(autoencoder_loss_pattern)

        masks = self._get_masks_from_sample(
            sample,
            *list(set(pattern['mask'] for pattern in loss_pattern + latent_loss_pattern + autoencoder_loss_pattern))
        )

        out, r_states_dynamics = self(inp[0][:, 0], *inp[1:], get_latent_states=True, **kwargs)

        ref, pred = self._get_reference_and_prediction(inp[0], out, loss_between_differences)
        loss = self._compute_loss(ref=ref, pred=pred, loss_pattern=loss_pattern, masks=masks)

        if latent_loss_pattern != () or autoencoder_loss_pattern != ():

            ae_kwargs = self._parse_ae_kwargs(kwargs)
            r_states_enc = self.state_enc(inp[0], *inp[4:], **ae_kwargs)

            if latent_loss_pattern != ():
                ref, pred = self._get_reference_and_prediction(
                    r_states_enc, r_states_dynamics, loss_between_differences
                )
                loss += self._compute_loss(ref=ref, pred=pred, loss_pattern=latent_loss_pattern, masks=masks)

            if autoencoder_loss_pattern != ():
                states_dec = self.state_dec(r_states_enc, *inp[4:], **ae_kwargs)
                ref, pred = self._get_reference_and_prediction(inp[0], states_dec, loss_between_differences)
                loss += self._compute_loss(ref=ref, pred=pred, loss_pattern=autoencoder_loss_pattern, masks=masks)

        loss.backward()
        return loss.detach()

    def _make_tbptt_training_iter(self, sample, loss_pattern, latent_loss_pattern, autoencoder_loss_pattern,
                                  loss_between_differences, tbptt_step, tbptt_tail, tbptt_loss_between_steps, **kwargs):

        if loss_between_differences:
            # TODO
            raise NotImplementedError('Loss between differences is not implemented for TBPTT.')

        args = self._get_attrs_from_sample(sample, *self._attrs, leave_attrs_device=self._attrs)
        states, rock, control, control_t, time = args[:5] #pylint:disable=unbalanced-tuple-unpacking
        args = args[5:]

        latent_loss_pattern = standardize_loss_pattern(latent_loss_pattern)
        autoencoder_loss_pattern = standardize_loss_pattern(autoencoder_loss_pattern)

        masks = self._get_masks_from_sample(sample, *[p['mask'] for p in loss_pattern])
        latent_masks = self._get_masks_from_sample(sample, *[p['mask'] for p in latent_loss_pattern])
        autoencoder_masks = self._get_masks_from_sample(sample, *[p['mask'] for p in autoencoder_loss_pattern])
        device = get_model_device(self)

        ae_kwargs = self._parse_ae_kwargs(kwargs)

        s_i = states[:, 0].to(device)
        rs_i = self.state_enc(s_i, *args, **ae_kwargs)
        buffer = [(None, rs_i)]
        rock = rock.to(device)

        targets = []
        outputs = []
        ae_outputs = []
        latent_outputs = []
        latent_targets = []

        loss = 0.
        n_steps_total = control.shape[1]
        if not tbptt_loss_between_steps:
            n_steps_backward = n_steps_total // tbptt_step
        else:
            n_steps_backward = n_steps_total

        for i in range(n_steps_total):

            rs_i = buffer[-1][1].detach()
            rs_i.requires_grad = True

            t_i = time[:, i:i+2]
            c_i, ct_i = self.get_control_subset(control, control_t, (t_i[0, 0], t_i[0, -1]))
            c_i, ct_i = c_i.to(device), ct_i.to(device)
            rc_i = self.control_enc(c_i, *args, **ae_kwargs)
            t_i = t_i.to(device)

            # TODO it is possible to do graph savings in a better way
            # https://stackoverflow.com/questions/50741344/pytorch-when-using-backward-how-can-i-retain-only-part-of-the-graph
            r_rock = self.params_enc(rock, *args, **ae_kwargs)

            next_rs = self._compute_dynamics((rs_i, r_rock, rc_i), ct_i, t_i, *args, **kwargs)
            next_rs = next_rs[:, 1]

            if loss_pattern or autoencoder_loss_pattern:
                targets.append(states[:, i+1].unsqueeze(1).to(device))
            if loss_pattern:
                next_s = self._decode(next_rs, *args, **ae_kwargs)
                outputs.append(next_s.unsqueeze(1))
            if autoencoder_loss_pattern or latent_loss_pattern:
                if targets:
                    next_rs_ae = self.state_enc(targets[-1], *args, **ae_kwargs)
                else:
                    next_rs_ae = self.state_enc(states[:, i+1].unsqueeze(1).to(device), *args, **ae_kwargs)
                if latent_loss_pattern:
                    latent_targets.append(next_rs_ae)
                    latent_outputs.append(next_rs.unsqueeze(1))
                if autoencoder_loss_pattern:
                    next_s_ae = self._decode(next_rs_ae, *args, **ae_kwargs)
                    ae_outputs.append(next_s_ae)

            _adjust_args_length(
                targets, outputs, ae_outputs, latent_targets, latent_outputs,
                max_len=(tbptt_step if tbptt_loss_between_steps else 1)
            )

            buffer.append((rs_i, next_rs))
            _adjust_args_length(buffer, max_len=tbptt_tail)

            if (i + 1) % tbptt_step == 0:

                for j in range(tbptt_step):
                    loss_i = torch.scalar_tensor(0).to(device)
                    if loss_pattern != ():
                        ref, pred = self._get_reference_and_prediction(
                            targets[-j - 1], outputs[-j - 1], loss_between_differences
                        )
                        loss_i += self._compute_loss(ref=ref, pred=pred, loss_pattern=loss_pattern, masks=masks)
                    if autoencoder_loss_pattern:
                        ref, pred = self._get_reference_and_prediction(
                            targets[-j - 1], ae_outputs[-j - 1], loss_between_differences
                        )
                        loss_i += self._compute_loss(
                            ref=ref, pred=pred, loss_pattern=autoencoder_loss_pattern, masks=autoencoder_masks
                        )
                    if latent_loss_pattern:
                        ref, pred = self._get_reference_and_prediction(
                            latent_targets[-j - 1], latent_outputs[-j - 1], loss_between_differences
                        )
                        loss_i += self._compute_loss(
                            ref=ref, pred=pred, loss_pattern=latent_loss_pattern, masks=latent_masks
                        )

                    loss_i /= n_steps_backward
                    loss += loss_i.detach()

                    retain_graph = (j != 0) or (tbptt_tail > tbptt_step)
                    loss_i.backward(retain_graph=retain_graph)

                    if not tbptt_loss_between_steps:
                        break

                for j in range(tbptt_tail - 1):
                    curr_grad = buffer[-j - 1][0].grad
                    retain_graph = j < tbptt_tail - tbptt_step - 1
                    buffer[-j - 2][1].backward(curr_grad, retain_graph=retain_graph)

                    # if we get all the way back to the initial state, stop
                    if buffer[-j - 2][0] is None:
                        break
        return loss

    @staticmethod
    def _get_reference_and_prediction(inp, out, loss_between_differences):
        if loss_between_differences:
            return inp[:, 1:] - inp[:, :-1], out[:, 1:] - out[:, :-1]
        return inp, out


def _adjust_args_length(*args, max_len):
    for arg in args:
        while len(arg) > max_len:
            del arg[0]
