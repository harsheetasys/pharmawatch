"""Lesson 3d: MGPS (Multi-item Gamma Poisson Shrinker, DuMouchel 1999) - the FDA's Bayesian method.

Idea: every drug-event pair has a true "reporting ratio" lambda (1 = nothing special).
We observe N reports where E were expected by chance:   N ~ Poisson(lambda * E).
Prior belief about lambda across ALL pairs = a mix of two gamma distributions
(one bump for "most pairs are boring", one for "some pairs are elevated").
We LEARN that prior from the counts themselves (no ground-truth labels used),
then for each pair compute:
   EBGM = shrunken best estimate of lambda
   EB05 = 5th percentile of lambda (the cautious score the paper ranked with; paper AUC 0.79)
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize
from scipy.special import digamma, expit, gammaln, logsumexp
from scipy.stats import gamma as gamma_dist
from sklearn.metrics import roc_auc_score

from benchmarks.bootstrap_auc import paired_bootstrap
from benchmarks.omop_prr import DATA_DIR


def log_nb(n, e, alpha, beta):
    """log P(N=n | E=e) when lambda ~ Gamma(alpha, rate=beta): a negative binomial."""
    return (gammaln(alpha + n) - gammaln(alpha) - gammaln(n + 1)
            + n * np.log(e / (e + beta)) + alpha * np.log(beta / (e + beta)))


def unpack(theta):
    a1, b1, a2, b2 = np.exp(theta[:4])          # keep alphas/betas positive
    w = expit(theta[4])                          # keep mixing weight in (0, 1)
    return a1, b1, a2, b2, w


def neg_log_likelihood(theta, n, e):
    a1, b1, a2, b2, w = unpack(theta)
    ll = logsumexp([np.log(w) + log_nb(n, e, a1, b1), np.log(1 - w) + log_nb(n, e, a2, b2)], axis=0)
    return -ll.sum()


def fit_prior(n, e):
    start = np.array([np.log(0.2), np.log(0.1), np.log(2.0), np.log(4.0), 0.0])   # DuMouchel's defaults
    best = minimize(neg_log_likelihood, start, args=(n, e), method="Nelder-Mead",
                    options={"maxiter": 20000, "xatol": 1e-6, "fatol": 1e-6})
    return unpack(best.x), best


def posterior(n, e, prior):
    """Posterior of lambda is again a 2-gamma mixture: weights q, shapes a+n, rates b+e."""
    a1, b1, a2, b2, w = prior
    l1 = np.log(w) + log_nb(n, e, a1, b1)
    l2 = np.log(1 - w) + log_nb(n, e, a2, b2)
    q = np.exp(l1 - np.logaddexp(l1, l2))
    return q, (a1 + n, b1 + e), (a2 + n, b2 + e)


def ebgm(n, e, prior):
    q, (s1, r1), (s2, r2) = posterior(n, e, prior)
    return np.exp(q * (digamma(s1) - np.log(r1)) + (1 - q) * (digamma(s2) - np.log(r2)))


def eb05(n, e, prior):
    q, (s1, r1), (s2, r2) = posterior(np.array([n]), np.array([e]), prior)
    q, s1, r1, s2, r2 = (float(x[0]) for x in (q, s1, r1, s2, r2))
    cdf = lambda lam: q * gamma_dist.cdf(lam, s1, scale=1 / r1) + (1 - q) * gamma_dist.cdf(lam, s2, scale=1 / r2) - 0.05
    return brentq(cdf, 1e-12, 1e6)


if __name__ == "__main__":
    res = pd.read_csv(DATA_DIR / "omop_scores_compared.csv").dropna(subset=["prr05"]).reset_index(drop=True)
    n, e = res["a"].to_numpy(float), res["expected"].to_numpy(float)

    prior, fit = fit_prior(n, e)
    a1, b1, a2, b2, w = prior
    print(f"Fitted prior (from counts only, no labels): {w:.2f} x Gamma({a1:.2f}, {b1:.2f}) + "
          f"{1 - w:.2f} x Gamma({a2:.2f}, {b2:.2f})   converged={fit.success}")

    res["ebgm"] = ebgm(n, e, prior)
    res["eb05"] = [eb05(ni, ei, prior) for ni, ei in zip(n, e)]
    res.to_csv(DATA_DIR / "omop_scores_compared.csv", index=False)

    for name, df in [("A) same 346 pairs", res[res.original_match]), ("B) all 391 pairs", res)]:
        aucs = {m: roc_auc_score(df.truth, df[m]) for m in ["prr05", "ic025", "ebgm", "eb05"]}
        print(f"\n{name}:  " + "  ".join(f"{m.upper()} {v:.3f}" for m, v in aucs.items()))
        for other in ["prr05", "ic025"]:
            obs, low, high, share = paired_bootstrap(df, "eb05", other)
            print(f"   EB05 - {other.upper():5s} = {obs:+.3f}   95% CI [{low:+.3f}, {high:+.3f}]   EB05 wins {share:.0%}")
    print("\nPaper on FAERS: PRR05 0.71 | MGPS EB05 0.79 | logistic regression 0.83")