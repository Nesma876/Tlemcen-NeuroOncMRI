import numpy as np

from evaluation.delong_test_and_xai_helpers import delong_roc_test
from fusion.fusion_base_strategies import fusion_moyenne_simple
from gat.patient_similarity_gat import build_patient_graph


def test_simple_fusion_average():
    out = fusion_moyenne_simple({"a": np.array([0.0, 1.0]), "b": np.array([0.5, 0.5])})
    np.testing.assert_allclose(out, np.array([0.25, 0.75]))


def test_patient_graph_shape_and_self_loops():
    graph = build_patient_graph(np.eye(4), k=1)
    assert graph.shape == (4, 4)
    assert np.all(np.diag(graph))


def test_delong_returns_finite_values():
    y = np.array([0, 0, 1, 1])
    a = np.array([0.1, 0.4, 0.8, 0.9])
    b = np.array([0.2, 0.3, 0.7, 0.95])
    diff, p = delong_roc_test(y, a, b)
    assert np.isfinite(diff)
    assert np.isfinite(p)
