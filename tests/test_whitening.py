import unittest
from tests._helpers import *
import config
from src.preprocessing import center
from src.whitening import whiten, covariance


class TestWhitening(unittest.TestCase):
    def setUp(self):
        self.X = np.array(config.MIXING_MATRIX) @ laplace_sources()
        self.Xc, _ = center(self.X)

    def test_identity_covariance_and_zero_mean(self):
        r = whiten(self.Xc)
        np.testing.assert_allclose(covariance(r.Z), np.eye(3), atol=1e-10)
        self.assertLess(np.max(np.abs(r.Z.mean(axis=1))), 1e-10)
        self.assertTrue(np.all(np.diff(r.eigvals) <= 0))  # descending

    def test_K_formula(self):
        r = whiten(self.Xc)
        C = covariance(self.Xc)
        np.testing.assert_allclose(r.K @ C @ r.K.T, np.eye(3), atol=1e-10)

    def test_degenerate_rejected(self):
        Xd = np.vstack([self.Xc[0], self.Xc[1], self.Xc[0] + self.Xc[1]])  # rank 2
        with self.assertRaises(ValueError):
            whiten(Xd)
        r = whiten(Xd, n_components=2)  # reducing components is allowed
        np.testing.assert_allclose(covariance(r.Z), np.eye(2), atol=1e-8)

    def test_uncentered_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            whiten(self.X + 10)
        bad = self.Xc.copy(); bad[1, 3] = np.inf
        with self.assertRaises(ValueError):
            whiten(bad)
        with self.assertRaises(ValueError):
            whiten(np.zeros((3, 100)))


if __name__ == "__main__":
    unittest.main()
