"""Property transfer models."""
import numpy as np

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors

class PropertiesTransfer(BaseEstimator):
    """Model to transfer rock properties using Nearest Neighbors algorithm."""
    def __init__(self):
        self._nn = None
        self._neigh_dist = None
        self._neigh_ind = None
        self._new_shape = None

    def fit(self, original_grid, new_grid, n_neighbors=5, normalize_vector=(1, 1, 1)):
        """Fit Nearest Neighbors model.

        Parameters
        ----------
        original_grid : geology.Grid
            Original grid.
        new_grid : geology.Grid
            New grid.
        n_neighbors : int, optional
            number of neighbors to use, by default 5.
        normalize_vector : list, optional
            vector to normalize distance, by default [1, 1, 1].
        """
        self._nn = NearestNeighbors(n_neighbors=n_neighbors)
        self._nn.fit(original_grid.cell_centroids.reshape(-1, 3) / np.asarray(normalize_vector))
        self._neigh_dist, self._neigh_ind = self._nn.kneighbors(
            new_grid.cell_centroids.reshape(-1, 3) / np.asarray(normalize_vector))
        self._new_shape = new_grid.dimens
        return self

    def dump(self, path):
        """Dump to a file (*.npz).

        Parameters
        ----------
        path : str or pathlib.Path
            Path to file, ``.npz`` extension will be appended to the file name if it is not
            already there.
        """
        np.savez(path, **{att: getattr(self, att) for att in ('_neigh_dist',
                                                              '_neigh_ind', '_new_shape')},
                 )
        return self

    def load(self, path):
        """Load from file.

        Parameters
        ----------
        path : str or pathlib.Path
            File path.
        """
        npzfile = np.load(path, allow_pickle=True)
        for att in npzfile.files:
            value = npzfile[att]
            setattr(self, att, value if (value.ndim > 0) or (value[()] is not None) else None)
        return self

    def predict(self, original_properties,
                aggr=lambda values, dist: np.average(values, axis=1, weights=1/dist)):
        """Transfer property values to the new grid.

        Parameters
        ----------
        original_properties : numpy.ndarray
            Properties on original grid.
        aggr : Callable, optional
            Function to aggregate nearest neighbors values, by default weighted average.

        Returns
        -------
        array : ndarray
            New values.
        """
        reshaped_properties = original_properties.reshape([-1] + list(original_properties.shape[-3:]))
        new_values = np.stack([
            aggr(prop.ravel()[self._neigh_ind], self._neigh_dist) for prop in reshaped_properties
        ])
        return new_values.reshape(list(original_properties.shape[:-3]) + list(self._new_shape))
