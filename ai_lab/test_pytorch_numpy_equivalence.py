import numpy as np

from experiment_digits_pytorch_nn import numpy_torch_equivalence


def test_pytorch_autograd_matches_numpy_backprop():
    rng = np.random.default_rng(10)
    x = rng.normal(size=(6, 64))
    y = np.array([0, 1, 2, 3, 4, 5])
    result = numpy_torch_equivalence(x, y)
    assert result["passed_under_1e-10"]
    assert result["probability_max_abs_difference"] < 1e-12
    assert result["loss_abs_difference"] < 1e-10
