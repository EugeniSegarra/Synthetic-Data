import unittest
import numpy as np
import synth_data as sd


class UtilsTestCase(unittest.TestCase):
    def test_rank_transform_shape(self):
        x = np.array([3.0, 1.0, 2.0, 4.0])
        y = sd.rank_transform(x)
        self.assertEqual(y.shape, x.shape)
        self.assertAlmostEqual(np.mean(y), 0.0, places=8)

    def test_ensure_positive_definite(self):
        A = np.array([[1.0, 2.0], [2.0, 1.0]])
        R = sd.ensure_positive_definite(A, use_higham=False)
        vals = np.linalg.eigvals(R)
        self.assertTrue(np.all(vals >= 0))


if __name__ == "__main__":
    unittest.main()
