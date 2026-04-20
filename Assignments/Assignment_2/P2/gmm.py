"""
P2(a): Gaussian Mixture Model via the Expectation-Maximization algorithm.

Implemented from scratch using NumPy only.
"""

import numpy as np


class GMM:
    """
    Full-covariance Gaussian Mixture Model fitted by EM.

    Parameters
    ----------
    n_components : int
        Number of mixture components K.
    max_iter : int
        Maximum EM iterations.
    tol : float
        Convergence threshold on log-likelihood change.
    reg_covar : float
        Regularisation added to diagonal of each covariance to ensure PSD.
    random_state : int or None
    """

    def __init__(self, n_components=1, max_iter=200, tol=1e-4,
                 reg_covar=1e-6, random_state=None):
        self.n_components = n_components
        self.max_iter     = max_iter
        self.tol          = tol
        self.reg_covar    = reg_covar
        self.random_state = random_state

    # ── Initialisation ──────────────────────────────────────────────────────

    def _init_params(self, X: np.ndarray) -> None:
        rng = np.random.default_rng(self.random_state)
        N, D = X.shape
        K = self.n_components

        # K-means++ seeding for stable initialisation
        idx = [int(rng.integers(N))]
        min_dists = np.sum((X - X[idx[0]]) ** 2, axis=1)   # (N,)
        for _ in range(K - 1):
            probs = min_dists / min_dists.sum()
            new_idx = int(rng.choice(N, p=probs))
            idx.append(new_idx)
            d = np.sum((X - X[new_idx]) ** 2, axis=1)
            min_dists = np.minimum(min_dists, d)

        self.means_  = X[idx].copy()                          # (K, D)
        self.covs_   = np.array([np.eye(D) for _ in range(K)])  # (K, D, D)
        self.weights_ = np.full(K, 1.0 / K)                  # (K,)

    # ── E-step ──────────────────────────────────────────────────────────────

    def _log_gaussian(self, X: np.ndarray, mean: np.ndarray,
                      cov: np.ndarray) -> np.ndarray:
        """Log N(x | mean, cov) for every row of X.  Shape: (N,)"""
        D = X.shape[1]
        diff = X - mean                              # (N, D)
        sign, log_det = np.linalg.slogdet(cov)
        if sign <= 0:
            # degenerate covariance — return very low log-prob
            return np.full(X.shape[0], -1e300)
        cov_inv = np.linalg.inv(cov)
        maha = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)  # (N,)
        return -0.5 * (D * np.log(2 * np.pi) + log_det + maha)

    def _e_step(self, X: np.ndarray):
        """
        Returns
        -------
        log_resp : (N, K)  — log responsibilities (unnormalised)
        log_likelihood : float
        """
        K = self.n_components
        N = X.shape[0]
        log_resp = np.empty((N, K))

        for k in range(K):
            log_resp[:, k] = (np.log(self.weights_[k] + 1e-300) +
                              self._log_gaussian(X, self.means_[k], self.covs_[k]))

        # log-sum-exp for numerical stability
        log_norm = log_resp.max(axis=1, keepdims=True)
        log_sum  = log_norm.squeeze() + np.log(
            np.exp(log_resp - log_norm).sum(axis=1))
        log_resp -= log_sum[:, None]                # normalise → log r_{nk}
        log_likelihood = log_sum.sum()
        return log_resp, log_likelihood

    # ── M-step ──────────────────────────────────────────────────────────────

    def _m_step(self, X: np.ndarray, log_resp: np.ndarray) -> None:
        N, D = X.shape
        K    = self.n_components
        resp = np.exp(log_resp)                     # (N, K)
        Nk   = resp.sum(axis=0)                     # (K,)

        self.weights_ = Nk / N

        for k in range(K):
            r_k = resp[:, k]                        # (N,)
            self.means_[k] = (r_k[:, None] * X).sum(axis=0) / Nk[k]

            diff = X - self.means_[k]               # (N, D)
            self.covs_[k] = (
                (r_k[:, None, None] * diff[:, :, None] *
                 diff[:, None, :]).sum(axis=0) / Nk[k]
            )
            # regularise diagonal
            self.covs_[k] += self.reg_covar * np.eye(D)

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "GMM":
        """Fit the model; returns self."""
        X = np.asarray(X, dtype=float)
        self._init_params(X)

        prev_ll = -np.inf
        for iteration in range(self.max_iter):
            log_resp, ll = self._e_step(X)
            self._m_step(X, log_resp)
            if abs(ll - prev_ll) < self.tol:
                break
            prev_ll = ll

        self.log_likelihood_ = ll
        self.n_iter_         = iteration + 1
        return self

    def score(self, X: np.ndarray) -> float:
        """Total log-likelihood of X under the fitted model."""
        X = np.asarray(X, dtype=float)
        _, ll = self._e_step(X)
        return float(ll)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior responsibilities, shape (N, K)."""
        X = np.asarray(X, dtype=float)
        log_resp, _ = self._e_step(X)
        return np.exp(log_resp)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Hard cluster assignment, shape (N,)."""
        return self.predict_proba(X).argmax(axis=1)
