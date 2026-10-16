import unittest
import warnings
from tests._helpers import *
import config
from src.pipeline import separate
from src.ica import fastica, symmetric_decorrelation
from src.evaluation import evaluate_separation, amari_index


class TestICA(unittest.TestCase):
    def setUp(self):
        self.S = laplace_sources()
        self.A = np.array(config.MIXING_MATRIX)
        self.X = self.A @ self.S

    def test_symmetric_decorrelation_orthogonal(self):
        W = symmetric_decorrelation(np.random.default_rng(0).standard_normal((3, 3)))
        np.testing.assert_allclose(W @ W.T, np.eye(3), atol=1e-10)

    def test_converges_and_separates(self):
        run = separate(self.X, seed=0, tol=1e-8, max_iter=500)
        self.assertTrue(run.ica.converged)
        ev = evaluate_separation(self.S, run.Y)
        for p in ev["per_source"]:
            self.assertGreater(p["abs_pearson"], 0.99)
        self.assertLess(amari_index(run.W_total @ self.A), 0.05)

    def test_all_nonlinearities(self):
        for nl in ("tanh", "cube", "gauss"):
            run = separate(self.X, seed=3, nonlinearity=nl, max_iter=800, tol=1e-7)
            ev = evaluate_separation(self.S, run.Y)
            self.assertGreater(ev["summary"]["mean_abs_pearson"], 0.98, nl)

    def test_seed_robustness(self):
        for seed in range(4):
            run = separate(self.X, seed=seed)
            ev = evaluate_separation(self.S, run.Y)
            self.assertGreater(ev["summary"]["mean_abs_pearson"], 0.99)

    def test_nonconvergence_reported(self):
        run_kwargs = dict(seed=0, max_iter=1, tol=1e-15)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            run = separate(self.X, **run_kwargs)
        self.assertFalse(run.ica.converged)
        self.assertTrue(any("did not converge" in str(x.message) for x in w))

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            fastica(np.random.default_rng(0).standard_normal((3, 1000)) * 5)  # not white
        with self.assertRaises(ValueError):
            fastica(np.full((3, 100), np.nan))
        with self.assertRaises(ValueError):
            separate(self.X, nonlinearity="bogus")


if __name__ == "__main__":
    unittest.main()
