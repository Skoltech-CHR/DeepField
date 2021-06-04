#pylint: disable=too-many-lines
"""Wells and WellSegment components."""
import numpy as np
import pandas as pd
from anytree import NodeMixin

from .rates import apply_perforations, calculate_cf
from .base_component import BaseComponent


class WellSegment(BaseComponent, NodeMixin):
    """Well's node.

    Parameters
    ----------
    name : str, optional
        Node's name.
    is_group : bool, optional
        Should a node represet a group of wells. Default to False.

    Attributes
    ----------
    is_group : bool
        Indicator of a group.
    is_main_branch : bool
        Indicator of a main branch.
    name : str
        Node's name.
    fullname : str
        Node's full name from root.
    """

    def __init__(self, *args, parent=None, children=None, name=None, is_group=False, **kwargs):
        super().__init__(*args, **kwargs)
        super().__setattr__('parent', parent)
        self._name = name
        self._is_group = is_group
        if children is not None:
            super().__setattr__('children', children)

    def copy(self):
        """Returns a deepcopy. Cached properties are not copied."""
        copy = super().copy()
        copy._name = self._name #pylint: disable=protected-access
        copy._is_group = self.is_group #pylint: disable=protected-access
        return copy

    @property
    def is_group(self):
        """Check that node is a group of wells."""
        return self._is_group

    @property
    def is_main_branch(self):
        """Check that node in a main well's branch."""
        return (not self.is_group) and (':' not in self.name)

    @property
    def name(self):
        """Node's name."""
        return self._name

    @property
    def fullname(self):
        """Full name from root."""
        return self.separator.join([node.name for node in self.path[1:]])

    @property
    def total_rates(self):
        """Total rates for the current node and all its branches."""
        columns = ['DATE', 'WOPR', 'WWPR', 'WGPR', 'WFGPR']
        if 'RESULTS' not in self:
            df = pd.DataFrame(columns=columns).set_index('DATE')
        else:
            df = self.results[[x for x in columns if x in self.results]].set_index('DATE')
        for node in self.children:
            df = df.add(node.total_rates.set_index('DATE'), fill_value=0)
        return df.reset_index()

    @property
    def cum_rates(self):
        """Cumulative rates for the current node and all its branches."""
        return self.total_rates.set_index('DATE').cumsum().reset_index()

    def perforated_indices(self, t=None):
        """Mask indicating perforated blocks."""
        if t is not None:
            raise NotImplementedError('Time parameter is not yet implemented.')
        if 'BLOCKS_INFO' in self.attributes:
            return (self.blocks_info['PERF_RATIO'] > 0).values
        return np.array([], dtype=np.bool)

    def perforated_blocks(self, t=None):
        """List of perforated blocks for current node."""
        if 'BLOCKS' in self.attributes:
            return self.blocks[self.perforated_indices(t)]
        return np.array([])

    def all_perforated_blocks(self, t=None):
        """List of perforated blocks for current node and all its descendants."""
        res = [node.perforated_blocks(t) for node in self.descendants]
        res.append(self.perforated_blocks(t))
        res = [x for x in res if len(x)]
        return np.unique(np.vstack(res), axis=0) if res else np.array([])

    def apply_perforations(self, current_date=None):
        """Open or close perforation intervals for given time interval.

        Parameters
        ----------
        segment : WellSegment
            Well's node.
        current_date : pandas.Timestamp
            Final date to open new perforations.

        Returns
        -------
        segment : WellSegment
            Segment with a blocks_info attribute that contains:
            - projections of welltrack in grid blocks
            - MD values for corresponding grid blocks
            - ratio of perforated part of the well for each grid block.
        """
        apply_perforations(self, current_date=current_date)
        return self

    def calculate_cf(self, rock, grid, beta=1, units='METRIC', cf_aggregation='sum'):
        """Calculate connection factor values for each grid block of a segment.

        Parameters
        ----------
        rock : Rock
            Rock component of geological model.
        grid : Grid
            Rock component of geological model.
        segment : WellSegment
            Well's node.
        beta : list or ndarray
            Additional multiplicator for productivity index of well.
        units : str, 'METRIC' or 'FIELD'
            Field units.
        cf_aggregation: str, 'sum' or 'eucl'
            The way of aggregating cf projection ('sum' - sum, 'eucl' - Euclid norm).

        Returns
        -------
        segment : WellSegment
            Segment with a blocks_info attribute extended with calculated connection
            factor values for all segment blocks.
        """
        calculate_cf(rock, grid, self, beta, units, cf_aggregation)
        return self
