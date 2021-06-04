"""Common tools for autoencoding modules."""
from .._base_nn_model import BaseModel


class SpatialAutoencoder(BaseModel):
    """Spatial autoencoder for States, Rock and Control."""
    def __init__(self, encoder, decoder, attr='states'):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attr = attr

    def forward(self, inp, *args, get_hidden_state=False, **kwargs):
        """
        Parameters
        ----------
        inp: nn.Module
        args: tuple
        get_hidden_state: bool
            If True, return hidden state.
        kwargs: dict

        Returns
        -------
        out: torch.Tensor
        hidden_state: torch.Tensor, optional
        """
        latent = self.encoder(inp, *args, **kwargs)
        inp = self.decoder(latent, *args, **kwargs)
        if not get_hidden_state:
            return inp
        return inp, latent

    def make_training_iter(self, sample, loss_pattern, **kwargs):
        """Make one training iter. Runs backward pass. Return differentiable loss.

        Parameters
        ----------
        sample: dict
        loss_pattern: dict
        kwargs: dict

        Returns
        -------
        loss
        """
        inp = self._get_attrs_from_sample(sample, self.attr)
        masks = self._get_masks_from_sample(sample, *[pattern['mask'] for pattern in loss_pattern])
        out = self(*inp, **kwargs)
        loss = self._compute_loss(
            ref=inp[0] if isinstance(inp, tuple) else inp,
            pred=out, loss_pattern=loss_pattern, masks=masks
        )
        loss.backward()
        return loss.detach()
