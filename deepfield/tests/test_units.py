"""Testsing module."""
import os
import pytest
import numpy as np
import pandas as pd

from ..field import Field, OrthogonalUniformGrid
from ..field.base_component import BaseComponent
from ..field.base_spatial import SpatialComponent
from ..field.getting_wellblocks import defining_wellblocks_vtk

from .data.test_wells import TEST_WELLS

@pytest.fixture(scope="module")
def tnav_model():
    """Load tNav test model."""
    test_path = os.path.dirname(os.path.realpath(__file__))
    tnav_path = os.path.join(test_path, 'data', 'tNav_test_model', 'TEST_MODEL.data')
    return Field(tnav_path, loglevel='ERROR').load()

@pytest.fixture(scope="module")
def hdf5_model():
    """Load HDF5 test model."""
    test_path = os.path.dirname(os.path.realpath(__file__))
    hdf5_path = os.path.join(test_path, 'data', 'hdf5_test_model', 'test_model.hdf5')
    return Field(hdf5_path, loglevel='ERROR').load()

@pytest.fixture(params=['tnav_model', 'hdf5_model'])
def model(request):
    """Returns model."""
    return request.getfixturevalue(request.param)

#pylint: disable=redefined-outer-name
class TestModelLoad:
    """Testing model load in tNav and HDF5 formats."""
    def test_content(self, model):
        """Testing components and attributes content."""
        assert set(model.components).issubset({'grid', 'rock', 'states', 'tables', 'wells', 'aquifers'})
        assert set(model.grid.attributes) == {'DIMENS', 'ZCORN', 'COORD', 'ACTNUM', 'MAPAXES'}
        assert set(model.rock.attributes) == {'PORO', }
        assert set(model.states.attributes) == {'PRESSURE', }
        assert len(model.wells.names) == len(TEST_WELLS)
        assert set(model.meta.keys()) == {'FLUIDS', 'UNITS', 'HUNITS', 'DATES',
                                          'START', 'TITLE', 'MODEL_TYPE'}

    def test_shape(self, model):
        """Testing data shape."""
        dimens = (2, 1, 6)
        assert np.all(model.grid.dimens == dimens)
        assert model.state.spatial
        assert np.all(model.rock.poro.shape == model.grid.dimens)
        assert np.all(model.grid.actnum.shape == model.grid.dimens)
        assert model.grid.zcorn.shape == dimens + (8, )
        assert np.all(model.grid.coord.shape == np.array([dimens[0] + 1, dimens[1] + 1, 6]))
        assert model.grid.mapaxes.shape == (6, )
        assert model.rock.poro.shape == dimens
        assert model.states.pressure.shape[1:] == dimens

    def test_dtype(self, model):
        """Testing data types."""
        def isfloat_32_or_64(x):
            return np.issubdtype(x.dtype, np.float64) or np.issubdtype(x.dtype, np.float32)

        assert np.issubdtype(model.grid.dimens.dtype, np.integer)
        assert np.issubdtype(model.grid.actnum.dtype, np.bool)
        assert isfloat_32_or_64(model.grid.zcorn)
        assert isfloat_32_or_64(model.grid.coord)
        assert isfloat_32_or_64(model.rock.poro)
        assert isfloat_32_or_64(model.states.pressure)


class TestPipeline():
    """Testing methods in pipelines."""
    def test_wells_pipeline(self, hdf5_model): #pylint: disable=redefined-outer-name
        """Testing wells processing."""
        model = hdf5_model.copy()
        model.wells.update({'no_welltrack': {'perf': pd.DataFrame()}})
        model.wells.drop_incomplete()
        assert 'no_welltrack' not in model.wells.names
        model.wells.get_wellblocks(model.grid)
        assert np.all(['BLOCKS' in node for node in model.wells])
        model.wells.drop_outside()
        assert len(model.wells.names) == 25
        assert min([node.blocks.size for node in model.wells]) > 0


class TestBaseComponent():
    """Testing BaseComponent."""

    def test_case(self):
        """Testing attrbutes are case insensitive."""
        bcomp = BaseComponent()
        bcomp.sample_attr = 1
        assert bcomp.SAMPLE_ATTR == 1
        assert bcomp.sample_attr == 1
        assert set(bcomp.attributes) == {'SAMPLE_ATTR', }

    def test_read_arrays(self):
        """Testing read arrays."""
        bcomp = BaseComponent()
        bcomp._read_buffer(['0 1 2 1*3\n2*4 5/'], attr='compr', compressed=True, dtype=int) #pylint:disable=protected-access
        bcomp._read_buffer(['0 1 2 3 4 4 5/'], attr='flat', compressed=False, dtype=int) #pylint:disable=protected-access

        assert isinstance(bcomp.flat, np.ndarray)
        assert isinstance(bcomp.compr, np.ndarray)
        assert bcomp.flat.shape == bcomp.compr.shape
        assert bcomp.flat.shape == (7, )
        assert np.all(bcomp.flat == bcomp.compr)

    def test_state(self):
        """Testing state."""
        bcomp = BaseComponent()
        bcomp.init_state(test=True)
        assert bcomp.state.test
        bcomp.set_state(test=False)
        assert ~bcomp.state.test


class TestSpatialComponent():
    """Testing SpatialComponent."""

    def test_ravel(self):
        """Testing ravel state."""
        data = np.arange(10).reshape(2, 5)
        comp = SpatialComponent()
        comp.arr = data
        comp.ravel()
        assert not comp.state.spatial
        assert comp.arr.shape == (10,)
        assert np.all(comp.arr == data.ravel(order='F'))


@pytest.fixture(scope="module")
def orth_grid():
    """Provides orthogonal uniform grid."""
    grid = OrthogonalUniformGrid(dimens=np.array([4, 6, 8]),
                                 dx=0.5, dy=1, dz=1.5,
                                 actnum=np.ones((4, 6, 8)))
    grid.set_state(spatial=True)
    return grid

class TestOrthogonalGrid():
    """Testing orthogonal uniform grids."""

    def test_setup(self, orth_grid): #pylint: disable=redefined-outer-name
        """Testing grid setup."""
        assert np.all(orth_grid.dimens == [4, 6, 8])
        assert np.all(orth_grid.cell_size == [0.5, 1, 1.5])

    def test_upscale(self, orth_grid): #pylint: disable=redefined-outer-name
        """Testing grid upscale and downscale methods."""
        upscaled = orth_grid.upscale(2)
        assert np.all(upscaled.cell_size == 2 * orth_grid.cell_size)
        assert np.all(upscaled.dimens == orth_grid.dimens / 2)
        assert np.all(upscaled.actnum == 1)
        assert upscaled.state.spatial
        downscaled = upscaled.downscale(2)
        assert np.all(downscaled.cell_size == orth_grid.cell_size)
        assert np.all(downscaled.dimens == orth_grid.dimens)
        assert downscaled.state.spatial


class TestCornerPointGrid():
    """Testing corner-point grids."""

    def test_setup(self, hdf5_model): #pylint: disable=redefined-outer-name
        """Testing grid setup."""
        grid = hdf5_model.grid
        assert np.all(grid.cell_volumes == 1)

    def test_upscale(self, hdf5_model): #pylint: disable=redefined-outer-name
        """Testing grid upscale and downscale methods."""
        grid = hdf5_model.grid
        upscaled = grid.upscale(factors=grid.dimens)
        assert np.all(upscaled.dimens == [1, 1, 1])
        assert np.all(upscaled.coord == [[[0., 0., 0., 0., 0., 6.],
                                          [0., 1., 0., 0., 1., 6.]],
                                         [[2., 0., 0., 2., 0., 6.],
                                          [2., 1., 0., 2., 1., 6.]]])
        assert np.all(upscaled.zcorn == [0., 0., 0., 0., 6., 6., 6., 6.])
        assert upscaled.state.spatial


class TestWellblocks():
    """Testing algorithm for defining wellblocks. """

    def test_algorithm(self):
        """Creating test wells and check blocks and intersections for every block."""
        grid = OrthogonalUniformGrid(dimens=np.array([2, 1, 6]),
                                     dx=1, dy=1, dz=1,
                                     actnum=np.ones((2, 1, 6)).astype(bool))
        grid.set_state(spatial=True)
        grid = grid.to_corner_point()

        grid.actnum[0, 0, 1] = False
        grid.actnum[0, 0, 4] = False
        grid.actnum[1, 0, 1] = False
        grid.actnum[1, 0, 4] = False

        grid.create_vtk_locator()

        for test_well in TEST_WELLS:
            output = defining_wellblocks_vtk(test_well['welltrack'], '1', grid,
                                             grid._vtk_locator, grid._cell_id_d) #pylint: disable=protected-access
            xyz_block, _, _, inters = output
            for i, block in enumerate(xyz_block):
                assert np.allclose(
                    test_well['blocks'][i], block), f"Error in defining blocks: {test_well['blocks'], xyz_block}"
                assert np.allclose(
                    test_well['inters'][i], inters[i]), f"Error in defining intersections {test_well['inters'], inters}"
