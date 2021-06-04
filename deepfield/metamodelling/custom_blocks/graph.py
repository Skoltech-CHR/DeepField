"""Custom blocks for modules based on graph convolutions."""
import math
import torch
from torch import nn


class RegularGraphConv(nn.Module):
    """Convolutional layer for variables defined on regular graphs."""
    def __init__(self, in_ch, out_ch, n_neighbours):
        """
        Parameters
        ----------
        in_ch: int
            Number of input channels.
        out_ch: int
            Number of output channels.
        n_neighbours: int
            Number of neighbours in the regular graph.
        """
        super().__init__()
        self.flattened_conv = nn.Linear(in_ch * n_neighbours, out_ch)
        self.fill_invalid = nn.Parameter(
            torch.empty(1, in_ch, 1), requires_grad=True
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize module's parameters."""
        bound = 1 / math.sqrt(self.fill_invalid.shape[0])
        nn.init.uniform_(self.fill_invalid, -bound, bound)

    def forward(self, inp, neighbours, distances=None):
        """
        Parameters
        ----------
        inp: torch.Tensor
            Input.
        neighbours: torch.Tensor
            Adjacency matrix with indices of neighbours for each graph node.
            Shape: [B, N, n_neighbours]
        distances: torch.Tensor
            Matrix of distances to the neighbouring nodes.
            Shape: [B, N, n_neighbours]

        Returns
        -------
        out: torch.Tensor

        """
        if distances is not None:
            raise NotImplementedError()
        addition = self.fill_invalid.expand(inp.shape[0], -1, -1)
        inp = torch.cat([inp, addition], dim=-1)
        out = []
        for b in range(inp.shape[0]):
            x_b = inp[b].transpose(0, 1)
            n_b = neighbours[b]
            neighbours_feature = x_b[n_b].reshape(x_b.shape[0] - 1, -1)
            out_b = self.flattened_conv(neighbours_feature)
            out.append(out_b.transpose(0, 1))
        out = torch.stack(out, dim=0)
        return out
