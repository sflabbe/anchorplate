import numpy as np


def minimum_norm_force_distribution(coords_mm, weights, ref_x_mm, ref_y_mm, force_n, mx_nmm, my_nmm):
    dx = coords_mm[:, 0] - ref_x_mm
    dy = coords_mm[:, 1] - ref_y_mm
    a = np.vstack([np.ones(coords_mm.shape[0]), dy, -dx])
    b = np.array([force_n, mx_nmm, my_nmm], dtype=float)
    awat = a @ (weights[:, None] * a.T)
    lam = np.linalg.solve(awat, b)
    return weights * (a.T @ lam)


def test_force_and_moment_recovery():
    coords = np.array([
        [75.0, 100.0],
        [75.0, 150.0],
        [75.0, 200.0],
        [225.0, 100.0],
        [225.0, 150.0],
        [225.0, 200.0],
    ])
    weights = np.array([25.0, 50.0, 25.0, 25.0, 50.0, 25.0])
    f = minimum_norm_force_distribution(coords, weights, 150.0, 150.0, 50000.0, 3.0e6, -2.0e6)
    dx = coords[:, 0] - 150.0
    dy = coords[:, 1] - 150.0
    assert abs(f.sum() - 50000.0) < 1e-8
    assert abs((dy * f).sum() - 3.0e6) < 1e-6
    assert abs((-dx * f).sum() + 2.0e6) < 1e-6
