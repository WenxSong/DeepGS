"""Coordinate preservation matters when averaging derivative libraries."""
import numpy as np
import pytest
import torch

from deepgs.library import CandidateLibrary, average_libraries, finite_difference_library, library_at_points


def library():
    return CandidateLibrary(np.arange(4.0)[:, None], np.ones((4, 2)), np.ones((4, 1)),
                            ["theta_z", "theta_zz"], np.array([.1, .1, .2, .2])[:, None],
                            np.array([0., 1., 0., 1.])[:, None])


@pytest.mark.parametrize("coordinate", ["depth", "time"])
def test_averaging_rejects_changed_row_coordinates(coordinate):
    first, second = library(), library()
    setattr(second, coordinate, getattr(second, coordinate)[::-1])
    with pytest.raises(ValueError, match="coordinates or row order"):
        average_libraries([first, second])


def test_averaging_checks_term_order_and_coordinate_presence():
    first, second = library(), library()
    second.names.reverse()
    with pytest.raises(ValueError, match="term names or order"):
        average_libraries([first, second])
    second = library()
    second.time = None
    with pytest.raises(ValueError, match="time coordinates"):
        average_libraries([first, second])
    with pytest.raises(ValueError, match="empty"):
        average_libraries([])


def test_aligned_average_keeps_coordinates_without_aliasing():
    first, second = library(), library()
    second.terms *= 3
    averaged = average_libraries([first, second])
    np.testing.assert_array_equal(averaged.terms, 2)
    np.testing.assert_array_equal(averaged.time, first.time)
    assert averaged.time is not first.time and averaged.depth is not first.depth


def test_fd_coordinates_follow_depth_major_rows_and_saved_archive(tmp_path):
    z, t = np.linspace(0, 1, 9), np.array([0., .2, .7, 1.])
    values = z[:, None] ** 3 + 2 * t[None, :] * z[:, None]
    result = finite_difference_library(values, z, t)
    zz, tt = np.meshgrid(z[2:-2], t[1:], indexing="ij")
    np.testing.assert_array_equal(result.depth[:, 0], zz.ravel())
    np.testing.assert_array_equal(result.time[:, 0], tt.ravel())
    np.testing.assert_allclose(result.theta[:, 0], (zz ** 3 + 2 * tt * zz).ravel())
    result.save(tmp_path / "library.npz")
    with np.load(tmp_path / "library.npz") as saved:
        np.testing.assert_array_equal(saved["time"], result.time)


def test_autodiff_coordinates_keep_arbitrary_point_order():
    class Field(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1., dtype=torch.float64))

        def forward(self, z, t):
            return self.scale * z ** 3 + 2 * t * z

    z = np.array([[.3], [.1], [.6]])
    t = np.array([[2.], [0.], [1.]])
    result = library_at_points(Field(), z, t, 3, batch_size=2)
    np.testing.assert_array_equal(result.time, t)
    np.testing.assert_array_equal(result.depth, z)
    np.testing.assert_allclose(result.theta[:, 0], (z ** 3 + 2 * t * z).ravel())


@pytest.mark.parametrize("coordinate", ["depth", "time"])
def test_coordinate_shape_and_finiteness_checked(coordinate):
    kwargs = dict(theta=np.ones((3, 1)), terms=np.ones((3, 1)), theta_t=np.ones((3, 1)), names=["theta_z"])
    with pytest.raises(ValueError, match=coordinate):
        CandidateLibrary(**kwargs, **{coordinate: np.array([[0.], [np.nan], [1.]])})
    with pytest.raises(ValueError, match=coordinate):
        CandidateLibrary(**kwargs, **{coordinate: np.ones((2, 1))})
