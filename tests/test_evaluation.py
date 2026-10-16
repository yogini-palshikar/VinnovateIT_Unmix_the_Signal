import unittest
from tests._helpers import *
from src.evaluation import (correlation_matrix, match_components, scale_align, si_sdr,
                            evaluate_separation)


class TestEvaluation(unittest.TestCase):
    def test_matching_recovers_permutation_and_sign(self):
        S = laplace_sources()
        perm = [2, 0, 1]
        Y = S[perm] * np.array([[-2.0], [0.5], [3.0]])
        corr = correlation_matrix(S, Y)
        assign = match_components(corr)
        for i in range(3):
            self.assertEqual(perm[assign[i]], i)
        ev = evaluate_separation(S, Y)
        for p in ev["per_source"]:
            self.assertAlmostEqual(p["abs_pearson"], 1.0, places=9)
            self.assertLess(p["nmse"], 1e-12)
            self.assertGreater(p["si_sdr_db"], 90)

    def test_scale_align_least_squares(self):
        s = laplace_sources()[0]
        a, alpha = scale_align(-4 * s, s)
        self.assertAlmostEqual(alpha, -0.25)
        np.testing.assert_allclose(a, s)
        z, al = scale_align(np.zeros(10), np.ones(10))
        self.assertEqual(al, 0.0)

    def test_si_sdr_known_value(self):
        s = laplace_sources()[0]
        s = s - s.mean()
        n = np.random.default_rng(5).standard_normal(s.size)
        n -= (n @ s) / (s @ s) * s  # orthogonal noise
        n = n / n.std() * s.std() * 0.1   # 20 dB below
        self.assertAlmostEqual(si_sdr(7 * (s + n), s), 20.0, delta=0.05)
        self.assertTrue(np.isnan(si_sdr(np.zeros_like(s), s)))

    def test_raw_output_not_modified(self):
        S = laplace_sources()
        Y = S[[1, 2, 0]] * 5
        Y0 = Y.copy()
        evaluate_separation(S, Y)
        np.testing.assert_array_equal(Y, Y0)

    def test_invalid_inputs(self):
        S = laplace_sources()
        with self.assertRaises(ValueError):
            correlation_matrix(S, S[:, :100])
        bad = S.copy(); bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            correlation_matrix(S, bad)
        with self.assertRaises(ValueError):
            match_components(np.ones((3, 2)))


if __name__ == "__main__":
    unittest.main()
