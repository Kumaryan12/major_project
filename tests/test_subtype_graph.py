import numpy as np

from mi_localization.distribution_kernel_model import KernelData, KernelModel
from mi_localization.subtype_graph import fit_dual, laplacian


def test_graph_dual_matches_direct_weighted_primal_system():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(10, 4))
    q = rng.normal(size=(5, 4))
    y = np.array([0, 1, 1, 2, 2, 0, 1, 0, 2, 0])
    data = KernelData(x, y, q, 3)
    graph = laplacian(['AMI', 'ALMI', 'HC'], [('AMI', 'ALMI')])
    alpha, strength = 0.7, 1.3
    centered = x - data.feature_mean
    penalty = np.eye(3) + strength * graph
    gram = centered.T @ (data.weights[:, None] * centered)
    target = centered.T @ (data.weights[:, None] * data.y_centered)
    system = np.kron(np.eye(3), gram) + alpha * np.kron(penalty.T, np.eye(x.shape[1]))
    primal = np.linalg.solve(system, target.ravel(order='F')).reshape(target.shape, order='F')
    expected = (q - data.feature_mean) @ primal + data.y_mean
    dual = fit_dual(data, alpha, strength, graph)
    _, cross, _ = data.kernel(0)
    np.testing.assert_allclose(cross @ dual + data.y_mean, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(dual[:, -1], KernelModel.fit(data, alpha, 0).dual[:, -1])


def test_zero_graph_penalty_matches_ordinary_ridge():
    rng = np.random.default_rng(31)
    data = KernelData(rng.normal(size=(12, 6)), np.arange(12) % 3, rng.normal(size=(4, 6)), 3)
    graph = laplacian(['IMI', 'ILMI', 'HC'], [('IMI', 'ILMI')])
    expected = KernelModel.fit(data, 1.0, 0).dual
    np.testing.assert_allclose(fit_dual(data, 1.0, 0., graph), expected, rtol=1e-10, atol=1e-12)
