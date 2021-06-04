"""Getting wellblocks."""
import numpy as np
import vtk
from .utils import length_segment

# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def defining_wellblocks_vtk(welltrack, well_name, grid, locator,
                            cell_id_d, axes=None, logger=None, rtol=1e-8):
    """The algorithm that solves raycating problem and fined all the necessary info

    Parameters
    ----------
    welltrack : array_like
        Well segment coordinates from welltrack.
    well_name: str
        Name of the current welltrack
    grid : class instance
        Basic grid class.
    locator: class instance
        vtk.vtkModifiedBSPTree
    cell_id_d: dict
        Mapping from index in final grid to the initial (x, y, z)
    axes: array_like
        Axes to project on
    logger: logger class instance
    rtol : scalar
        np.allclose tolerance. Default 1e-8.

    Returns
    -------
    wellblocks : array_like
        Coordinates of grid blocks crossed by segment.
    h_well : array_like
        Projections of segment in every block which it crosses.
    blocks_md : array_like
        Coordinates of grid blocks derived out of welltrack file and their MD values.
    cells_track_points: array_like
        Array of intersection points for every crossed block

    """
    if axes is None:
        axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    cells_track = []
    cells_track_points = []

    w_track = welltrack[:, :3].copy()
    mds = welltrack[:, 3].copy()
    is_inside = False # at the start of the iteration tells if we were inside the grid on the prev iteration
    ever_inside = False # tells if the well have ever entered the grid
    prev_points_cash = [w_track[0]]

    cum_length = mds[0]
    cum_length_d = {}
    cum_length_proj = {}
    i = 0

    def get_projections(start, end, axes):
        u = np.array(end) - np.array(start)
        return np.abs(axes @ u) / np.linalg.norm(axes, axis=1)

    def logger_print(msg):
        if logger is not None:
            logger.warning(msg)

    # pylint: disable=too-many-nested-blocks
    while i < len(w_track) - 1:

        new_cell = None
        points_vtk_intersection = vtk.vtkPoints()
        points_vtk_intersection.SetDataTypeToDouble()
        cells_vtk_intersection = vtk.vtkIdList()

        code = locator.IntersectWithLine(w_track[i], w_track[i+1], 1e-10,
                                         points_vtk_intersection, cells_vtk_intersection)

        if code:
            points_vtk_intersection_data = points_vtk_intersection.GetData()
            ncells_vtk_intersection = cells_vtk_intersection.GetNumberOfIds()
            npoints_vtk_intersection = points_vtk_intersection_data.GetNumberOfTuples()

            # need to count distinct points and cells
            points_list = [] # порядок важен
            for idx in range(npoints_vtk_intersection):
                point_tup = points_vtk_intersection_data.GetTuple3(idx)
                if len(points_list) == 0:
                    points_list.append(point_tup)
                else:
                    if point_tup != points_list[-1]:
                        points_list.append(point_tup)

            cells_set = set()
            for idx in range(ncells_vtk_intersection):
                edge_id = cells_vtk_intersection.GetId(idx)
                cell_id = edge_id//6
                cells_set.add(cell_id)

            # Let's find the next intersection
            new_code = 0
            k = i + 1
            while not new_code and k < len(w_track) - 1:

                next_points_vtk_intersection = vtk.vtkPoints()
                next_points_vtk_intersection.SetDataTypeToDouble()
                next_cells_vtk_intersection = vtk.vtkIdList()
                new_code = locator.IntersectWithLine(w_track[k], w_track[k+1], 1e-10,
                                                     next_points_vtk_intersection,
                                                     next_cells_vtk_intersection)

                next_points_vtk_intersection_data = next_points_vtk_intersection.GetData()
                next_npoints_vtk_intersection = next_points_vtk_intersection_data.GetNumberOfTuples()

                # need to count distinct points and cells
                next_points_list = [] # порядок важен
                for idx in range(next_npoints_vtk_intersection):
                    point_tup = next_points_vtk_intersection_data.GetTuple3(idx)
                    if len(next_points_list) == 0:
                        next_points_list.append(point_tup)
                    else:
                        if point_tup != next_points_list[-1]:
                            next_points_list.append(point_tup)

                k += 1

            if len(points_list) == 1:
                l1 = length_segment(w_track[i], np.array(points_list[0]))
                l2 = length_segment(np.array(points_list[0]), w_track[i+1])
                l3 = length_segment(w_track[i], w_track[i+1])
                k1 = l1/l3
                k2 = l2/l3
                md_l = mds[i+1] - mds[i]
                cum_length += k1*md_l
                cum_length_d[tuple(points_list[0])] = cum_length
                cum_length_proj[tuple(points_list[0])] = get_projections(w_track[i], w_track[i+1], axes)*k1
                cum_length += k2*md_l
                cum_length_d[tuple(w_track[i+1])] = cum_length
                cum_length_proj[tuple(w_track[i+1])] = get_projections(w_track[i], w_track[i+1], axes)*k2
                if len(cells_set) == 1:
                    #this situation can be only if we are entrering or leaving (in some way) the grid
                    if not ever_inside:
                        ever_inside = True # change the value of ever_inside
                        new_cell = list(cells_set)[0]
                        cells_track.append(new_cell) # first cell
                        cells_track_points.append([])
                        if np.allclose(np.array(points_list[0]), w_track[i], rtol=rtol):
                            if grid.point_inside_cell(w_track[i+1], cell_id_d[new_cell]):
                                cells_track_points[-1].extend([w_track[i], w_track[i+1]])
                                is_inside = True
                            else:
                                cells_track_points[-1].extend([w_track[i]])
                                is_inside = False
                        elif np.allclose(np.array(points_list[0]), w_track[i+1], rtol=rtol):
                            if grid.point_inside_cell(w_track[i], cell_id_d[new_cell]):
                                if len(prev_points_cash) > 1:
                                    cells_track_points[-1].extend(prev_points_cash)
                                    cells_track_points[-1].extend([w_track[i+1]])
                                else:
                                    cells_track_points[-1].extend([w_track[i], w_track[i+1]])
                                prev_points_cash = []
                                is_inside = False
                            else:
                                cells_track_points[-1].extend([w_track[i+1]])
                                is_inside = True
                        else:
                            if grid.point_inside_cell(w_track[i], cell_id_d[new_cell]):
                                if len(prev_points_cash) > 1:
                                    cells_track_points[-1].extend(prev_points_cash)
                                    cells_track_points[-1].extend([np.array(points_list[0])])
                                else:
                                    cells_track_points[-1].extend([w_track[i], np.array(points_list[0])])
                                prev_points_cash = []
                                is_inside = False
                            else:
                                if grid.point_inside_cell(w_track[i+1], cell_id_d[new_cell]):
                                    cells_track_points[-1].extend([np.array(points_list[0]), w_track[i+1]])
                                    is_inside = True
                                else:
                                    is_inside = False

                    else: # ever_inside = True
                        if is_inside: # знаем, что w_track[i] внутри
                            if np.allclose(np.array(points_list[0]), w_track[i], rtol=rtol):
                                cells_track_points[-1].extend([w_track[i+1]])
                            else:
                                is_inside = False #leaving the grid
                                cells_track_points[-1].extend([np.array(points_list[0])])

                        else: # мы вне грида, но уже были в нем когда-то, is_inside = False
                            if np.allclose(np.array(points_list[0]), w_track[i+1], atol=rtol):
                                is_inside = True
                                new_cell = list(cells_set)[0]
                                if new_cell != cells_track[-1]:
                                    cells_track.append(new_cell)
                                    cells_track_points.append([])
                                cells_track_points[-1].extend([w_track[i+1]])
                            else:
                                new_cell = list(cells_set)[0]
                                if grid.point_inside_cell(w_track[i+1], cell_id_d[new_cell]):
                                    is_inside = True
                                    if new_cell != cells_track[-1]:
                                        cells_track.append(new_cell)
                                        cells_track_points.append([])
                                    cells_track_points[-1].extend([np.array(points_list[0]), w_track[i+1]])

                else: # 1 точка, больше одной ячейки,
                    if ever_inside:
                        # it's not the beginng of the well
                        if is_inside:
                            # сейчас внутри - w_track[i] внутри грида
                            if not new_code:
                                if np.allclose(np.array(points_list[0]), w_track[i], rtol=1e-8):
                                    cells_track_points[-1].extend([w_track[i+1]])
                                elif np.allclose(np.array(points_list[0]), w_track[i+1], rtol=1e-8):
                                    assert i+1 == len(w_track)-1
                                    cells_track_points[-1].extend([w_track[i+1]])
                                    # is_inside does not matter anymore
                                else:
                                    found_correct = False
                                    for amb_cell in list(cells_set):
                                        if grid.point_inside_cell(w_track[i+1], cell_id_d[amb_cell]):
                                            if not found_correct:
                                                found_correct = True
                                                cells_track_points[-1].extend([np.array(points_list[0])])
                                                if amb_cell != cells_track[-1]:
                                                    cells_track.append(amb_cell)
                                                    cells_track_points.append([])
                                                cells_track_points[-1].extend([np.array(points_list[0]), w_track[i+1]])
                                            else:
                                                logger_print(f"In well {well_name} some point can not be uniquely assigned to the cell") # pylint: disable=line-too-long
                                    if not found_correct:
                                        cells_track_points[-1].extend([np.array(points_list[0])])
                                        is_inside = False

                            else:
                                # new_code = True
                                if np.allclose(np.array(points_list[0]), w_track[i], rtol=rtol):
                                    # уже нашли на пред итерации
                                    cells_track_points[-1].extend([w_track[i+1]])
                                elif np.allclose(np.array(points_list[0]), w_track[i+1], rtol=rtol):
                                    found_correct = False
                                    if len(next_points_list) == 1:
                                        point_to_check = w_track[i+2]
                                    else:
                                        point_to_check = np.asarray([(w_track[i+1][j] + next_points_list[1][j])/2. for j in range(3)]) # pylint: disable=line-too-long
                                    for amb_cell in list(cells_set):
                                        if grid.point_inside_cell(point_to_check, cell_id_d[amb_cell]):
                                            if not found_correct:
                                                found_correct = True
                                                cells_track_points[-1].extend([w_track[i+1]])
                                                if amb_cell != cells_track[-1]:
                                                    cells_track.append(amb_cell)
                                                    cells_track_points.append([])
                                                cells_track_points[-1].extend([w_track[i+1]])
                                            else:
                                                logger_print(f"In well {well_name} some point can not be uniquely assigned to the cell") # pylint: disable=line-too-long
                                    if not found_correct:
                                        is_inside = False
                                        cells_track_points[-1].extend([w_track[i+1]])
                                else:
                                    found_correct = False
                                    for amb_cell in list(cells_set):
                                        if grid.point_inside_cell(w_track[i+1], cell_id_d[amb_cell]):
                                            if not found_correct:
                                                found_correct = True
                                                cells_track_points[-1].extend([np.array(points_list[0])])
                                                if amb_cell != cells_track[-1]:
                                                    cells_track.append(amb_cell)
                                                    cells_track_points.append([])
                                                cells_track_points[-1].extend([np.array(points_list[0]), w_track[i+1]])
                                            else:
                                                logger_print(f"In well {well_name} some point can not be uniquely assigned to the cell") # pylint: disable=line-too-long
                                    if not found_correct:
                                        is_inside = False
                                        cells_track_points[-1].extend([np.array(points_list[0])])

                        else:
                            if np.allclose(np.array(points_list[0]), w_track[i+1], rtol=rtol):
                                if i+1 != len(w_track)-1:
                                    found_correct = False
                                    if len(next_points_list) == 1:
                                        point_to_check = w_track[i+2]
                                    else:
                                        point_to_check = np.asarray([(w_track[i+1][j] + next_points_list[1][j])/2. for j in range(3)]) # pylint: disable=line-too-long
                                    for amb_cell in list(cells_set):
                                        if grid.point_inside_cell(point_to_check, cell_id_d[amb_cell]):
                                            if not found_correct:
                                                found_correct = True
                                                if amb_cell != cells_track[-1]:
                                                    cells_track.append(amb_cell)
                                                    cells_track_points.append([])
                                                cells_track_points[-1].extend([w_track[i+1]])
                                            else:
                                                logger_print(f"In well {well_name} some point can not be uniquely assigned to the cell") # pylint: disable=line-too-long

                            else:
                                found_correct = False
                                for amb_cell in list(cells_set):
                                    if grid.point_inside_cell(w_track[i+1], cell_id_d[amb_cell]):
                                        if not found_correct:
                                            found_correct = True
                                            is_inside = True
                                            if amb_cell != cells_track[-1]:
                                                cells_track.append(amb_cell)
                                                cells_track_points.append([])
                                            if np.allclose(np.array(points_list[0]), w_track[i], rtol=1e-8):
                                                cells_track_points[-1].extend([w_track[i+1]])
                                            else:
                                                cells_track_points[-1].extend([np.array(points_list[0]), w_track[i+1]])
                                        else:
                                            logger_print(f"In well {well_name} some point can not be uniquely assigned to the cell") # pylint: disable=line-too-long

                    else:
                        ever_inside = True
                        # assert is_inside == False
                        if np.allclose(np.array(points_list[0]), w_track[i], rtol=rtol):
                            assert i == 0
                            found_correct = False
                            for amb_cell in list(cells_set):
                                if grid.point_inside_cell(w_track[i+1], cell_id_d[amb_cell]):
                                    if not found_correct:
                                        found_correct = True
                                        is_inside = True
                                        cells_track.append(amb_cell)
                                        cells_track_points.append([])
                                        cells_track_points[-1].extend([w_track[i], w_track[i+1]])
                                    else:
                                        logger_print(f"In well {well_name} some point can not be uniquely assigned to the cell") # pylint: disable=line-too-long
                            if not found_correct:
                                is_inside = False
                        elif np.allclose(np.array(points_list[0]), w_track[i+1], rtol=rtol):
                            found_correct = False
                            for amb_cell in list(cells_set):
                                if grid.point_inside_cell(w_track[i], cell_id_d[amb_cell]):
                                    if not found_correct:
                                        found_correct = True
                                        is_inside = True
                                        cells_track.append(amb_cell)
                                        cells_track_points.append([])
                                        if len(prev_points_cash) > 1:
                                            cells_track_points[-1].extend(prev_points_cash)
                                            cells_track_points[-1].extend([w_track[i], w_track[i+1]])
                                        else:
                                            cells_track_points[-1].extend([w_track[i], w_track[i+1]])
                                        prev_points_cash = []
                                    else:
                                        logger_print(f"In well {well_name} some point can not be uniquely assigned to the cell") # pylint: disable=line-too-long
                            if new_code:
                                assert i+1 != len(w_track)-1
                                found_correct = False
                                if len(next_points_list) == 1:
                                    point_to_check = w_track[i+2]
                                else:
                                    point_to_check = np.asarray([(w_track[i+1][j] + next_points_list[1][j])/2. for j in range(3)]) # pylint: disable=line-too-long
                                for amb_cell in list(cells_set):
                                    if grid.point_inside_cell(point_to_check, cell_id_d[amb_cell]):
                                        if not found_correct:
                                            found_correct = True
                                            cells_track.append(amb_cell)
                                            cells_track_points.append([])
                                            cells_track_points[-1].extend([w_track[i+1]])
                                        else:
                                            logger_print(f"In well {well_name} some point can not be uniquely assigned to the cell") # pylint: disable=line-too-long
                                if not found_correct:
                                    is_inside = False
                        else:
                            is_inside = True
                            found_correct = False
                            for amb_cell in list(cells_set):
                                if grid.point_inside_cell(w_track[i], cell_id_d[amb_cell]):
                                    if not found_correct:
                                        found_correct = True
                                        cells_track.append(amb_cell)
                                        cells_track_points.append([])
                                        if len(prev_points_cash) > 1:
                                            cells_track_points[-1].extend(prev_points_cash)
                                            cells_track_points[-1].extend([w_track[i], np.array(points_list[0])])
                                        else:
                                            cells_track_points[-1].extend([w_track[i], np.array(points_list[0])])
                                        prev_points_cash = []
                                    else:
                                        logger_print(f"In well {well_name} some point can not be uniquely assigned to the cell") # pylint: disable=line-too-long

                            found_correct = False
                            for amb_cell in list(cells_set):
                                if grid.point_inside_cell(w_track[i+1], cell_id_d[amb_cell]):
                                    if not found_correct:
                                        found_correct = True
                                        cells_track.append(amb_cell)
                                        cells_track_points.append([])
                                        cells_track_points[-1].extend([np.array(points_list[0]), w_track[i+1]])
                                    else:
                                        logger_print(f"In well {well_name} some point can not be uniquely assigned to the cell") # pylint: disable=line-too-long
                            if not found_correct:
                                is_inside = False

            else: # точек > 1, ячеек >= 1
                start = w_track[i]
                for l in range(len(points_list)): # pylint: disable=consider-using-enumerate
                    if l == len(points_list)-1:
                        end = w_track[i+1]
                    else:
                        end = np.asarray([(points_list[l][j] + points_list[l+1][j])/2. for j in range(3)])

                    l1 = length_segment(start, np.array(points_list[l]))
                    l2 = length_segment(np.array(points_list[l]), end)
                    l3 = length_segment(w_track[i], w_track[i+1])
                    k1 = l1/l3
                    k2 = l2/l3
                    md_l = mds[i+1] - mds[i]
                    cum_length += k1*md_l
                    cum_length_d[tuple(points_list[l])] = cum_length
                    cum_length_proj[tuple(points_list[l])] = get_projections(w_track[i], w_track[i+1], axes)*k1
                    cum_length += k2*md_l
                    cum_length_d[tuple(end)] = cum_length
                    cum_length_proj[tuple(end)] = get_projections(w_track[i], w_track[i+1], axes)*k2

                    if not ever_inside:
                        # work with the start point
                        if not np.allclose(start, np.array(points_list[l]), rtol=rtol):
                            found_cells = []
                            for amb_cell in list(cells_set):
                                if grid.point_inside_cell(start, cell_id_d[amb_cell]):
                                    found_cells.append(amb_cell)
                            if len(found_cells) == 1:
                                amb_cell = found_cells[0]
                            elif len(found_cells) > 1:
                                amb_cell = found_cells[0]
                            else:
                                amb_cell = None

                            if amb_cell is not None:
                                ever_inside = True
                                is_inside = True
                                cells_track.append(amb_cell)
                                cells_track_points.append([])
                                if len(prev_points_cash) > 1:
                                    cells_track_points[-1].extend(prev_points_cash)
                                    cells_track_points[-1].extend([np.array(points_list[l])])
                                    prev_points_cash = []
                                else:
                                    cells_track_points[-1].extend([start, np.array(points_list[l])])

                        # work with the end point
                        if not np.allclose(end, np.array(points_list[l]), rtol=rtol):
                            found_cells = []
                            for amb_cell in list(cells_set):
                                if grid.point_inside_cell(end, cell_id_d[amb_cell]):
                                    found_cells.append(amb_cell)

                            if len(found_cells) == 1:
                                amb_cell = found_cells[0]
                            elif len(found_cells) > 1:
                                amb_cell = found_cells[0]
                            else:
                                amb_cell = None
                                is_inside = False

                            if amb_cell is not None:
                                ever_inside = True
                                is_inside = True
                                cells_track.append(amb_cell)
                                cells_track_points.append([])
                                cells_track_points[-1].extend([np.array(points_list[l]), end])

                    else:
                        #ever inside true
                        is_inside = True
                        # work with the start point
                        if not np.allclose(start, np.array(points_list[l]), rtol=rtol):
                            found_cells = []
                            for amb_cell in list(cells_set):
                                if grid.point_inside_cell(start, cell_id_d[amb_cell]):
                                    found_cells.append(amb_cell)

                            if len(found_cells) == 1:
                                amb_cell = found_cells[0]
                            elif len(found_cells) > 1:
                                if cells_track[-1] in found_cells:
                                    amb_cell = cells_track[-1]
                                else:
                                    logger_print(f"In well {well_name} some point can not be uniquely assigned to the cell") # pylint: disable=line-too-long
                                    amb_cell = found_cells[0]
                            else:
                                amb_cell = None

                            if amb_cell is not None:
                                is_inside = True
                                cells_track_points[-1].extend([np.array(points_list[l])])

                        # work with the end point
                        if not np.allclose(end, np.array(points_list[l]), rtol=rtol):
                            found_cells = []
                            for amb_cell in list(cells_set):
                                if grid.point_inside_cell(end, cell_id_d[amb_cell]):
                                    found_cells.append(amb_cell)

                            if len(found_cells) == 1:
                                amb_cell = found_cells[0]
                            elif len(found_cells) > 1:
                                if cells_track[-1] in found_cells:
                                    amb_cell = cells_track[-1]
                                else:
                                    amb_cell = found_cells[0]
                            else:
                                amb_cell = None
                                is_inside = False

                            if amb_cell is not None:
                                is_inside = True
                                if amb_cell != cells_track[-1]:
                                    cells_track.append(amb_cell)
                                    cells_track_points.append([])
                                if np.allclose(start, np.array(points_list[l]), rtol=1e-8):
                                    cells_track_points[-1].extend([end])
                                else:
                                    cells_track_points[-1].extend([np.array(points_list[l]), end])
                        else:
                            if i+1 != len(w_track)-1:
                                if new_code:
                                    if len(next_points_list) == 1:
                                        point_to_check = w_track[i+2]
                                    else:
                                        point_to_check = np.asarray([(end[j] + next_points_list[1][j])/2. for j in range(3)]) # pylint: disable=line-too-long
                                else:
                                    point_to_check = w_track[i+2]
                                found_cells = []
                                for amb_cell in list(cells_set):
                                    if grid.point_inside_cell(point_to_check, cell_id_d[amb_cell]):
                                        found_cells.append(amb_cell)

                                if len(found_cells) == 1:
                                    amb_cell = found_cells[0]
                                elif len(found_cells) > 1:
                                    if cells_track[-1] in found_cells:
                                        amb_cell = cells_track[-1]
                                    else:
                                        amb_cell = found_cells[0]
                                else:
                                    amb_cell = None
                                    is_inside = False

                                if amb_cell is not None:
                                    is_inside = True
                                    if amb_cell != cells_track[-1]:
                                        cells_track.append(amb_cell)
                                        cells_track_points.append([])
                                    cells_track_points[-1].extend([end])

                    start = end

        else: # отрезок w_track[i] w_track[i+1] не имеет пересечений
            l3 = length_segment(w_track[i], w_track[i+1])
            md_l = mds[i+1] - mds[i]
            cum_length += md_l
            cum_length_d[tuple(w_track[i+1])] = cum_length
            cum_length_proj[tuple(w_track[i+1])] = get_projections(w_track[i], w_track[i+1], axes)*l3/md_l
            if is_inside:
                # мы все в той же ячейке, добавляем конец отрезка
                cells_track_points[-1].extend([w_track[i+1]])
            else:
                if not ever_inside:
                    # is_inside = Flase, ever_inside = False
                    # либо мы до сих пор вне грида и не входили в него
                    # либо мы до сих пор внутри одной ячейки и там и начали
                    prev_points_cash.append(w_track[i+1])

        i += 1

    xyz_block = []
    welltr_block_md = []
    for i in range(len(cells_track)): # pylint: disable=consider-using-enumerate
        xyz_block.append(cell_id_d[cells_track[i]])
        welltr_block_md.append(cum_length_d[tuple(cells_track_points[i][-1])])

    h_well = []
    for i in range(len(cells_track)):
        points_arr = cells_track_points[i]
        proj_s = np.array([0., 0., 0.])
        for j in range(1, len(points_arr)):
            proj_s += cum_length_proj[tuple(points_arr[j])]
        h_well.append(proj_s)
    return np.asarray(xyz_block), h_well, np.asarray(welltr_block_md), np.asarray(cells_track_points)

# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
def find_first_entering_point(welltrack, grid, locator,
                              cell_id_d):
    """The algorithm that solves raycating problem and fined all the necessary info

    Parameters
    ----------
    welltrack : array_like
        Well segment coordinates from welltrack.
    well_name: str
        Name of the current welltrack
    grid : class instance
        Basic grid class.
    locator: class instance
        vtk.vtkModifiedBSPTree
    cell_id_d: dict
        Mapping from index in final grid to the initial (x, y, z)
    axes: array_like
        Axes to project on
    logger: logger class instance

    Returns
    -------
    wellblocks : array_like
        Coordinates of grid blocks crossed by segment.
    h_well : array_like
        Projections of segment in every block which it crosses.
    blocks_md : array_like
        Coordinates of grid blocks derived out of welltrack file and their MD values.
    cells_track_points: array_like
        Array of intersection points for every crossed block

    """

    w_track = welltrack[:, :3].copy()
    i = 0
    code = False

    while not code and i < len(w_track)-1:
        points_vtk_intersection = vtk.vtkPoints()
        points_vtk_intersection.SetDataTypeToDouble()
        cells_vtk_intersection = vtk.vtkIdList()

        code = locator.IntersectWithLine(w_track[i], w_track[i+1], 1e-10,
                                         points_vtk_intersection, cells_vtk_intersection)

        if code:
            points_vtk_intersection_data = points_vtk_intersection.GetData()
            ncells_vtk_intersection = cells_vtk_intersection.GetNumberOfIds()
            npoints_vtk_intersection = points_vtk_intersection_data.GetNumberOfTuples()

            # need to count distinct points and cells
            points_list = [] # порядок важен
            for idx in range(npoints_vtk_intersection):
                point_tup = points_vtk_intersection_data.GetTuple3(idx)
                if len(points_list) == 0:
                    points_list.append(point_tup)
                else:
                    if point_tup != points_list[-1]:
                        points_list.append(point_tup)

            cells_set = set()
            for idx in range(ncells_vtk_intersection):
                edge_id = cells_vtk_intersection.GetId(idx)
                cell_id = edge_id//6
                cells_set.add(cell_id)

            found_cells = False
            for amb_cell in list(cells_set):
                if grid.point_inside_cell(w_track[i], cell_id_d[amb_cell]):
                    found_cells = True
                    break
            if not found_cells:
                return i, points_list[0]

            return None, None

        i += 1

    return None, None

def defining_wellblocks_compdat(compdat):
    """Get wellblocks from `COMPDAT` table.

    Parameters
    ----------
    compdat : pandas.DataFrame
        `COMPDAT` table.

    Returns
    -------
    numpy.ndarray
        Block indices.
    """
    i = []
    j = []
    k = []
    for _, row in compdat.iterrows():
        k_row = list(range(int(row['K1']-1), int(row['K2'])))
        k += k_row
        i += [int(row['I'])-1] * len(k_row)
        j += [int(row['J'])-1] * len(k_row)
    return np.array(list(set((a, b, c) for a, b, c in zip(i, j, k))))
