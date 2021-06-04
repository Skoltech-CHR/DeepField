"""Neural Ordinary Differential Equations dynamics."""
import torch
from torch import nn
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint

from .._base_nn_model import BaseModel
from ..custom_blocks.wrappers import CheckpointWrapper
from ..utils import get_model_device, find_best_match_indices


# pylint: disable=not-callable


# pylint: disable=invalid-name
class checkpointed_cls:
    """Decorates a module with `CheckpointWrapper`."""
    def __init__(self, cls):
        self.cls = cls

    def __call__(self, *args, **kwargs):
        return CheckpointWrapper(self.cls(*args, **kwargs))


class NeuralDifferentialEquation(BaseModel):
    """Module for modelling dynamics in the form of Neural Ordinary Differential Equation."""
    _attrs = ('states', 'rock', 'control', ('masks', 'control_t'), ('masks', 'time'))

    def __init__(self, time_derivative_module, time_scale=30.5):
        """
        Parameters
        ----------
        time_derivative_module: nn.Module
            Module, which computes the derivative of states over time, given params and instantaneous control.
        time_scale: float, optional
            Scales time axis. Default: 30.5
        """
        super().__init__()
        self._integrator = _Integrator(time_derivative_module, time_scale)
        self._checkpointed_integrator = _CheckpointedIntegrator(time_derivative_module, time_scale)
        self.last_n_integration_iter = 0

    def forward(self, init_state, params, control, control_t, t,
                *der_args, max_integration_timestep=100, solver='rk4', rtol=1e-4,
                atol=1e-4, use_adjoint=False, max_seq_len=None,
                use_outer_checkpointing=False, use_inner_checkpointing=True, **der_kwargs):
        """
        Parameters
        ----------
        init_state: torch.Tensor
            Initial state. Shape: (B, C_s, *dims)
        params: torch.Tensor
            Params tensor (rock mostly). Shape: (B, C_p, *dims)
        control: torch.Tensor
            Spatial control tensor. Shape: (B, T, C_c, *dims)
        control_t: torch.Tensor
            Tensor holding a sequence of time points for which control is given.
            Each time must be strictly larger than the previous time. Shape: (B, T_c)
        t: torch.Tensor
            Tensor holding a sequence of time points for which to solve for states.
            The initial time point (i.e. time point of the init_state) should be the first
            element of this sequence, and each time must be strictly larger than the previous time.
            Shape: (B, T)
        der_args: tuple
            Any additional args passed to the dynamics module.
        max_integration_timestep: int
            Maximal timestep can be made internally by the solver.
        solver: str
            Name of the solver to use. From the list:
            [
                'explicit_adams',
                'fixed_adams',
                'adams',
                'tsit5',
                'dopri5',
                'euler',
                'midpoint',
                'rk4',
                'adaptive_heun',
            ]
        rtol: float
            Relative tolerance.
        atol: float
            Absolute tolerance.
        use_adjoint: bool
            If True, use Adjoint method at the backward pass.
        max_seq_len: int
            Maximal lenght of the sequence, that can be passed into single forward pass.
            If required solution has a greater length, several forward passes will be done.
        use_outer_checkpointing: bool
            If True, decorates the Integrator class with CheckpointWrapper. Default: False.
        use_inner_checkpointing: bool
            If True, wraps individual forward passes of length `max_seq_len` with CheckpointWrapper. Default: True.
        der_kwargs: dict
            Any additional named arguments passed to the `time_derivative_module`.

        Returns
        -------
        solution: torch.Tensor
            Predicted states. Shape: (B, T, C_s, *dims)
        """
        kwargs = dict(solver=solver, rtol=rtol, atol=atol, max_integration_timestep=max_integration_timestep,
                      use_adjoint=use_adjoint, derivative_args=der_args, derivative_kwargs=der_kwargs)

        use_checkpointing = der_kwargs.get('use_checkpointing', False)
        if use_checkpointing:
            kwargs['use_inner_checkpointing'] = use_inner_checkpointing
        # TODO will work only if timesteps across batch are equal
        assert torch.all(t - t[:1] == 0), "Solution timesteps should be equal for batch items."
        t = t[0]
        assert torch.all(control_t - control_t[:1] == 0), "Control timesteps should be equal for batch items."
        control_t = control_t[0]

        if use_checkpointing:
            self._checkpointed_integrator.wrapped_module.set_kwargs(**kwargs, max_seq_len=max_seq_len)
            states = self._checkpointed_integrator(
                init_state, params, control, control_t, t, use_checkpointing=use_outer_checkpointing
            )
            self.last_n_integration_iter = self._checkpointed_integrator.wrapped_module.last_n_integration_iter
        else:
            self._integrator.set_kwargs(**kwargs)
            states = self._integrator(init_state, params, control, control_t, t)
            self.last_n_integration_iter = self._integrator.last_n_integration_iter
        return states

    def make_training_iter(self, sample, loss_pattern, loss_between_differences=False,
                           tbptt_step=None, tbptt_tail=None, tbptt_loss_between_steps=True, **kwargs):

        if (tbptt_step is not None and tbptt_tail is None) or (tbptt_step is None and tbptt_tail is not None):
            raise ValueError('If TBPTT is assumed to be used, both `k1` and `k2` should be specified.')
        if tbptt_step is None and tbptt_tail is None:
            return self._make_training_iter(sample, loss_pattern, loss_between_differences, **kwargs)
        return self._make_tbptt_training_iter(
            sample, loss_pattern, loss_between_differences, tbptt_step, tbptt_tail, tbptt_loss_between_steps, **kwargs
        )

    def _make_training_iter(self, sample, loss_pattern, loss_between_differences, **kwargs):
        """Make one training iteration. Runs backward pass. Return loss of the prediction.

        Parameters
        ----------
        sample: dict
            Sample from a dataset. Should contain 'states', 'control', 'rock', 'time' and all the required masks.
        loss_pattern: tuple
            Loss pattern between ground truth and predicted states.
        loss_between_differences: bool
            If True, loss is computed not between states, but rather between "gradients" (differences) of states.
        kwargs: dict
            Any additional named arguments passed to the `time_derivative_module`.

        Returns
        -------
        loss
            Detached loss.
        """
        inp = self._get_attrs_from_sample(sample, *self._attrs, leave_attrs_device=('control', ))
        masks = self._get_masks_from_sample(sample, *[pattern['mask'] for pattern in loss_pattern])
        out = self(inp[0][:, 0], *inp[1:], **kwargs)
        ref, pred = self._get_reference_and_prediction(inp[0], out, loss_between_differences)
        loss = self._compute_loss(ref=ref, pred=pred, loss_pattern=loss_pattern, masks=masks)
        loss.backward()
        return loss.detach()

    def _make_tbptt_training_iter(self, sample, loss_pattern, loss_between_differences,
                                  tbptt_step, tbptt_tail, tbptt_loss_between_steps, **kwargs):

        if loss_between_differences:
            # TODO
            raise NotImplementedError('Loss between differences is not implemented for TBPTT.')
        args = self._get_attrs_from_sample(sample, *self._attrs, leave_attrs_device=self._attrs)
        states, rock, control, time = args[:4] #pylint: disable=unbalanced-tuple-unpacking
        args = args[4:]
        masks = self._get_masks_from_sample(sample, *[pattern['mask'] for pattern in loss_pattern])
        device = get_model_device(self)

        s_i = states[:, 0].to(device)
        buffer = [(None, s_i)]

        outputs = []
        targets = []

        loss = 0.
        n_steps_total = control.shape[1]
        if not tbptt_loss_between_steps:
            n_steps_backward = n_steps_total // tbptt_step
        else:
            n_steps_backward = n_steps_total

        for i in range(n_steps_total):

            s_i = buffer[-1][1].detach()
            s_i.requires_grad = True

            c_i = control[:, i].unsqueeze(1).to(device)
            t_i = time[:, i:i+2].to(device)

            next_s = self(s_i, rock, c_i, t_i, *args, **kwargs)[:, 1]

            outputs.append(next_s.unsqueeze(1))
            targets.append(states[:, i+1].unsqueeze(1).to(device))

            while len(outputs) > tbptt_step:
                # Delete stuff that is too old
                del outputs[0]
                del targets[0]

            buffer.append((s_i, next_s))
            while len(buffer) > tbptt_tail:
                # Delete stuff that is too old
                del buffer[0]

            if (i + 1) % tbptt_step == 0:

                for j in range(tbptt_step):
                    ref, pred = self._get_reference_and_prediction(
                        targets[-j - 1], outputs[-j - 1], loss_between_differences
                    )
                    loss_i = self._compute_loss(ref=ref, pred=pred, loss_pattern=loss_pattern, masks=masks)
                    loss_i /= n_steps_backward
                    loss += loss_i.detach()

                    retain_graph = (j != 0) or (tbptt_tail > tbptt_step)
                    loss_i.backward(retain_graph=retain_graph)

                    if not tbptt_loss_between_steps:
                        break

                for j in range(tbptt_tail - 1):
                    # if we get all the way back to the initial state, stop
                    if buffer[-j - 2][0] is None:
                        break

                    curr_grad = buffer[-j - 1][0].grad
                    retain_graph = j < tbptt_tail - tbptt_step - 1
                    buffer[-j - 2][1].backward(curr_grad, retain_graph=retain_graph)
        return loss

    @staticmethod
    def _get_reference_and_prediction(inp, out, loss_between_differences):
        if loss_between_differences:
            return inp[:, 1:] - inp[:, :-1], out[:, 1:] - out[:, :-1]
        return inp, out


@checkpointed_cls
class _CheckpointedIntegrator(nn.Module):
    """Class which wraps an `_Integrator` forward passes with CheckpointWrapper.
    Also, it provides capability to split long sequences into separate smaller forward passes."""
    def __init__(self, derivative_module, time_scale):
        """
        Parameters
        ----------
        derivative_module: nn.Module
            Module, which computes the derivative of states over time, given params and instantaneous control.
        time_scale: float
            Scales time axis.
        """
        super().__init__()
        self._integrator = CheckpointWrapper(
            _Integrator(derivative_module, time_scale)
        )
        self.max_seq_len = None
        self.last_n_integration_iter = 0

    def forward(self, init_state, params, control, control_t, t):
        """
        Parameters
        ----------
        init_state: torch.Tensor
            Initial state. Shape: (B, C_s, *dims).
        params: torch.Tensor
            Params tensor (rock mostly). Shape: (B, C_p, *dims).
        control: torch.Tensor
            Spatial control tensor. Shape: (B, T_c, C_c, *dims).
        control_t: torch.Tensor
            Tensor holding a sequence of time points for which control is given.
            Each time must be strictly larger than the previous time. Shape: (T_c, )
        t: torch.Tensor
            Tensor holding a sequence of time points for which to solve for states.
            The initial time point (i.e. time point of the init_state) should be the first
            element of this sequence, and each time must be strictly larger than the previous time.
            Shape: (T, )

        Returns
        -------
        solution: torch.Tensor
            Predicted states. Shape: (B, T, C_s, *dims).
        """
        states = [init_state.unsqueeze(1)]
        s_prev = init_state
        t_pieces = self._get_t_pieces(t)

        for t_piece in t_pieces:
            s_piece = self._integrator(s_prev, params, control, control_t, t_piece, use_checkpointing=True)

            if t_piece[-1] in t:
                states.append(s_piece[:, 1:])
            else:
                states.append(s_piece[:, 1:-1])
            s_prev = s_piece[:, -1]

        states = torch.cat(states, dim=1)
        self.last_n_integration_iter = self._integrator.wrapped_module.last_n_integration_iter
        return states

    def _get_t_pieces(self, t):
        """Split time sequence into pieces."""
        if self.max_seq_len:
            return self._get_t_pieces_from_max_seq_len(t)
        return self._get_t_pieces_from_solution_t(t)

    def _get_t_pieces_from_max_seq_len(self, t):
        """Split time sequence into pieces by the given maximal length."""
        pieces = []
        integration_t = self._integrator.wrapped_module.get_integration_t(t)
        edge_ind = torch.arange(
            start=0, end=integration_t.shape[0] + self.max_seq_len, step=self.max_seq_len
        )
        for left, right in zip(edge_ind[:-1], edge_ind[1:]):
            piece = integration_t[left: right + 1]
            if piece.shape[0] == 1:
                continue
            inner_piece = piece[1:-1]
            if inner_piece.shape[0]:
                in_t = torch.any(t.view(-1, 1) == inner_piece.view(1, -1), dim=0)
                inner_piece = inner_piece[in_t]
            if inner_piece.shape[0]:
                piece = torch.cat([
                    piece[0].unsqueeze(0), inner_piece, piece[-1].unsqueeze(0)
                ])
            else:
                piece = torch.stack([piece[0], piece[-1]])
            pieces.append(piece)
        return pieces

    @staticmethod
    def _get_t_pieces_from_solution_t(t):
        """Split time sequence into pieces by the points where solution is mandatory."""
        pieces = []
        for left, right in zip(t[:-1], t[1:]):
            piece = torch.stack([left, right])
            pieces.append(piece)
        return pieces

    def set_kwargs(self, **kwargs):
        """Set kwargs for the wrapped `_Integrator`. instance."""
        self.max_seq_len = kwargs.pop('max_seq_len', None)
        use_inner_checkpointing = kwargs.pop('use_inner_checkpointing', True)
        if 'use_checkpointing' in kwargs['derivative_kwargs']:
            kwargs['derivative_kwargs']['use_checkpointing'] = use_inner_checkpointing
        self._integrator.wrapped_module.set_kwargs(**kwargs)


class _Integrator(nn.Module):
    """Integrator for a Neural Ordinary Differential Equation."""
    kwarg_keys = (
        'max_integration_timestep', 'solver', 'rtol', 'atol',
        'use_adjoint', 'derivative_args', 'derivative_kwargs'
        )

    def __init__(self, derivative_module, time_scale):
        """
        Parameters
        ----------
        derivative_module: nn.Module
            Module, which computes the derivative of states over time, given params and instantaneous control.
        time_scale: float
            Scales time axis.
        """
        super().__init__()
        self.derivative_module = _TimeDerivativeModule(
            derivative_module
        )
        self.time_scale = time_scale
        self.last_n_integration_iter = 0
        for key in self.kwarg_keys:
            setattr(self, key, None)

    def set_kwargs(self, **kwargs):
        """Assigns kwargs to self for following implicit use."""
        for key, value in kwargs.items():
            if key not in self.kwarg_keys:
                raise ValueError('Unexpected key found: %s' % key)
            setattr(self, key, value)

    def forward(self, init_state, params, control, control_t, t):
        """
        Parameters
        ----------
        init_state: torch.Tensor
            Initial state. Shape: (B, C_s, *dims)
        params: torch.Tensor
            Params tensor (rock mostly). Shape: (B, C_p, *dims)
        control: torch.Tensor
            Spatial control tensor. Shape: (B, T, C_c, *dims)
        control: torch.Tensor
            Spatial control tensor. Shape: (B, T_c, C_c, *dims)
        control_t: torch.Tensor
            Tensor holding a sequence of time points for which control is given.
            Each time must be strictly larger than the previous time. Shape: (T_c, )
        t: torch.Tensor
            Tensor holding a sequence of time points for which to solve for states.
            The initial time point (i.e. time point of the init_state) should be the first
            element of this sequence, and each time must be strictly larger than the previous time.
            Shape: (T, )

        Returns
        -------
        solution: torch.Tensor
            Predicted states. Shape: (B, T, C_s, *dims)
        """
        # TODO time for solution and time for control should be different
        self.derivative_module.set_params(params)
        self.derivative_module.set_control(control, control_t / self.time_scale)
        self.derivative_module.set_additional_forward_args(self.derivative_args, self.derivative_kwargs)

        integration_t = self.get_integration_t(t)

        integrator = odeint if not self.use_adjoint else odeint_adjoint
        states = integrator(
            self.derivative_module, init_state, integration_t / self.time_scale,
            method=self.solver, rtol=self.rtol, atol=self.atol
        ).transpose(1, 0)
        self.last_n_integration_iter = self.derivative_module.finish_integration()

        solution_indices = find_best_match_indices(search_for=t, search_in=integration_t)
        states = states[:, solution_indices]
        return states

    def get_integration_t(self, solution_t):
        """Get the set of timesteps, passed to the integrator.
        It can be longer than the `solution_t` due to the bounded length of solver's step."""
        t = [solution_t[0].unsqueeze(0)]
        for prev, cur in zip(solution_t[:-1], solution_t[1:]):
            prev, cur = float(prev), float(cur)
            piece = torch.arange(prev, cur, step=self.max_integration_timestep)
            if prev == cur or piece[-1] != cur:
                piece = torch.cat([piece, torch.tensor([cur])])
            t.append(piece[1:].to(solution_t.device))
        return torch.cat(t)


class _TimeDerivativeModule(nn.Module):
    """Module which computes derivative of states over time.
    Services as a transition to `odeint` function from `torchdiffeq`.
    Adds a control and parameters to the forward pass implicitly
    (since `odeint` assume no control or additional parameters used)."""
    def __init__(self, layers):
        """
        Parameters
        ----------
        layers: nn.Module
            Module, which computes the derivative of states over time, given params and instantaneous control.
        """
        super().__init__()
        self.layers = layers
        self._n_integration_iter = 0
        self._params = None
        self._control = None
        self._additional_forward_args = None

    def set_params(self, params):
        """Set params for following forward passes."""
        self._params = params

    def set_control(self, control, t):
        """Set control for following forward passes."""
        self._control = {'t': t, 'values': control}

    def set_additional_forward_args(self, args, kwargs):
        """Set additional args and kwargs for following forward passes."""
        self._additional_forward_args = {'args': args, 'kwargs': kwargs}

    def _get_control_at_t(self, t):
        """Get control corresponding to the given timestep from the sequence of controls."""
        i = find_best_match_indices(search_for=t, search_in=self._control['t'], less_or_equal=True).squeeze()
        if i >= self._control['values'].shape[1]:
            i = self._control['values'].shape[1] - 1
        return self._control['values'][:, i].to(self._control['t'].device)

    def finish_integration(self):
        """Clear the cash after the integration is finished.
        Return number of forward passes made during the integration."""
        n_integration_iter = self._n_integration_iter
        self._n_integration_iter = 0
        return n_integration_iter

    def forward(self, t, state):
        """Compute time-derivative of states at given state and timestep, and implicitly set control and parameters."""
        self._n_integration_iter += 1
        return self._inner_loop_forward(
            state, self._params, self._get_control_at_t(t),
            *self._additional_forward_args['args'], **self._additional_forward_args['kwargs']
        )

    def _inner_loop_forward(self, state, params, control, *args, **kwargs):
        """Compute time-derivative ds/dt."""
        out = torch.cat([state, params, control], dim=1)
        out = self.layers(out, *args, **kwargs)
        return out
