"""Group samples by water content, optionally within each depth, and normalize the regression system."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .library import CandidateLibrary


@dataclass
class GroupedSystem:
    """X[g] has a constant in column 0. Scales convert normalized coefficients to physical ones."""

    X: list[np.ndarray]
    y: list[np.ndarray]
    theta_centers: np.ndarray
    depth_centers: np.ndarray | None
    x_scale: np.ndarray  # (n_groups, n_features)
    y_scale: np.ndarray  # (n_groups,)
    feature_names: list[str]
    normalization: str = "global"

    @property
    def n_features(self) -> int:
        return self.X[0].shape[1]

    @property
    def n_groups(self) -> int:
        return len(self.X)

    @property
    def n_samples(self) -> int:
        return sum(block.shape[0] for block in self.X)

    def to_physical(self, W: np.ndarray) -> np.ndarray:
        """W has shape (n_features, n_groups). Each group uses its own scale."""
        return W / self.x_scale.T * self.y_scale


def theta_windows(theta_sorted: np.ndarray, win_size: float, min_group_size: int) -> list[np.ndarray]:
    """Non-overlapping windows covering every sorted sample exactly once."""
    windows, start, n = [], 0, theta_sorted.size
    while start < n:
        end = int(np.searchsorted(theta_sorted, theta_sorted[start] + win_size, side="right"))
        if end - start < min_group_size and start + min_group_size < n:
            end = start + min_group_size
        if n - end < min_group_size:
            end = n
        windows.append(np.arange(start, end))
        start = end
    return windows


def merge_tails(windows: list[np.ndarray], fraction: float) -> list[np.ndarray]:
    """Merge the driest and wettest fraction of windows, which often hold unreliable derivatives."""
    if fraction <= 0 or len(windows) < 3:
        return windows
    low = int(np.ceil(fraction * len(windows)))
    high = int(np.floor((1 - fraction) * len(windows)))
    return [np.concatenate(windows[:low]), *windows[low:high], np.concatenate(windows[high:])]


def _column_norms(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=0)
    return np.where(norms > 1e-12, norms, 1.0)


def build_system(library: CandidateLibrary, win_size: float = 0.001, min_group_size: int = 100,
                 normalization: str = "global", by_depth: bool = False, merge_tail: float = 0.0) -> GroupedSystem:
    """Assemble the grouped system.

    normalization is 'global' (unit norm over all samples, so groups keep their relative energy)
    or 'group' (unit norm within each group, so groups weigh equally). The AIC epsilon means
    different things under the two choices and must be set together with it.
    """
    if normalization not in ("global", "group"):
        raise ValueError(f"Unknown normalization: {normalization}")
    if by_depth and library.depth is None:
        raise ValueError("Grouping by depth needs a library with depth values")

    theta = library.theta[:, 0]
    x = np.column_stack((np.ones(theta.size), library.terms))
    y = library.theta_t.copy()
    if normalization == "global":
        global_x, global_y = _column_norms(x), float(_column_norms(y)[0])
        x, y = x / global_x, y / global_y

    labels = np.round(library.depth[:, 0], 9) if by_depth else np.zeros(theta.size)
    Xs, ys, theta_centers, depth_centers, x_scale, y_scale = [], [], [], [], [], []
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        order = members[np.argsort(theta[members], kind="stable")]
        for window in merge_tails(theta_windows(theta[order], win_size, min_group_size), merge_tail):
            index = order[window]
            block_x, block_y = x[index], y[index]
            if normalization == "group":
                scale_x, scale_y = _column_norms(block_x), float(_column_norms(block_y)[0])
                block_x, block_y = block_x / scale_x, block_y / scale_y
            else:
                scale_x, scale_y = global_x, global_y
            Xs.append(block_x)
            ys.append(block_y)
            theta_centers.append(theta[index].mean())
            depth_centers.append(label)
            x_scale.append(scale_x)
            y_scale.append(scale_y)

    if len(Xs) < 2:
        raise ValueError("Not enough groups; decrease win_size or min_group_size")
    return GroupedSystem(Xs, ys, np.asarray(theta_centers), np.asarray(depth_centers) if by_depth else None,
                         np.asarray(x_scale), np.asarray(y_scale), library.feature_names, normalization)
