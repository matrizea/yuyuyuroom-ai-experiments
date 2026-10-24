import numpy as np

from experiment_digits_numpy_nn import TwoLayerNet, gradient_check


def test_softmax_rows_sum_to_one():
    model = TwoLayerNet(4, 5, 3, seed=1)
    probabilities, _ = model.forward(np.ones((6, 4)))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-12)


def test_backprop_matches_finite_difference():
    rng = np.random.default_rng(2)
    model = TwoLayerNet(4, 5, 3, seed=3)
    x = rng.normal(size=(5, 4))
    y = np.array([0, 1, 2, 1, 0])
    result = gradient_check(model, x, y, checks=40)
    assert result["passed_under_1e-5"]
    assert result["max_relative_error"] < 1e-5
