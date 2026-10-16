import unittest
from tests._helpers import *
from src.preprocessing import center, apply_centering


class TestCentering(unittest.TestCase):
    def test_zero_mean(self):
        X = laplace_sources() + np.array([[5.0], [-3.0], [100.0]])
        Xc, m = center(X)
        self.assertLess(np.max(np.abs(Xc.mean(axis=1))), 1e-10)
        np.testing.assert_allclose(Xc + m, X)
        np.testing.assert_allclose(apply_centering(X, m), Xc)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            center(np.array([1.0, 2.0, 3.0]))
        bad = laplace_sources(); bad[0, 5] = np.nan
        with self.assertRaises(ValueError):
            center(bad)


if __name__ == "__main__":
    unittest.main()
