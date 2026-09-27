import unittest
import torch
from experiment_fashion_cnn import SmallCNN


class CNNTests(unittest.TestCase):
    def test_shape_parameters_and_gradients(self):
        model = SmallCNN()
        output = model(torch.zeros(4, 1, 28, 28))
        self.assertEqual(tuple(output.shape), (4, 10))
        self.assertEqual(sum(p.numel() for p in model.parameters()), 105866)
        loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 1, 2, 3]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
