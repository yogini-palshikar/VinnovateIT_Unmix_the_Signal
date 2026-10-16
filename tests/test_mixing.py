import unittest
from tests._helpers import *
import config
from src.mixing import validate_mixing_matrix, mix_sources, add_noise_at_snr


class TestMixing(unittest.TestCase):
    def test_main_matrix_full_rank(self):
        info = validate_mixing_matrix(np.array(config.MIXING_MATRIX), 3)
        self.assertEqual(info["rank"], 3)
        self.assertLess(info["condition_number"], 100)

    def test_singular_rejected(self):
        with self.assertRaises(ValueError):
            validate_mixing_matrix(np.array([[1, 2, 3], [2, 4, 6], [0, 1, 1.0]]))

    def test_underdetermined_rejected(self):
        with self.assertRaises(ValueError):
            validate_mixing_matrix(np.ones((2, 3)))

    def test_mix_shape_and_value(self):
        S = laplace_sources()
        A = np.array(config.MIXING_MATRIX)
        X = mix_sources(S, A)
        self.assertEqual(X.shape, S.shape)
        np.testing.assert_allclose(X[0], A[0] @ S)

    def test_snr_is_accurate(self):
        X = mix_sources(laplace_sources(), np.array(config.MIXING_MATRIX))
        Xn, meas = add_noise_at_snr(X, 10, np.random.default_rng(0))
        np.testing.assert_allclose(meas, 10, atol=0.1)
        self.assertEqual(Xn.shape, X.shape)


if __name__ == "__main__":
    unittest.main()
