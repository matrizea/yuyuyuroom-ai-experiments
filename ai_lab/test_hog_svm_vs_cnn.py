"""Small CPU-only checks for the HOG/CNN comparison's fixed configuration."""
import unittest
import numpy as np
import torch
from experiment_hog_svm_vs_cnn import SmallCNN, extract_hog, per_class_accuracy

class HOGComparisonTests(unittest.TestCase):
    def test_hog_dimensions_and_dtype(self):
        features=extract_hog(np.zeros((2,28,28),dtype=np.uint8))
        self.assertEqual(features.shape,(2,1296))
        self.assertEqual(features.dtype,np.float32)
        self.assertTrue(np.isfinite(features).all())

    def test_model_shape_and_parameter_count(self):
        model=SmallCNN()
        self.assertEqual(sum(p.numel() for p in model.parameters()),105866)
        self.assertEqual(tuple(model(torch.zeros(2,1,28,28)).shape),(2,10))

    def test_confusion_matrix_counts(self):
        labels=np.arange(10)
        matrix,accuracy=per_class_accuracy(labels,labels)
        self.assertEqual(int(matrix.sum()),10)
        np.testing.assert_array_equal(accuracy,np.ones(10))

if __name__=='__main__':
    unittest.main()
