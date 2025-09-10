from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Union, List
import numpy as np

# ================================================================
# Base class
# ================================================================
class MonotonicModel(ABC):
    """
    Training DB format:
      train_db: Dict[case_id, {"set": set(...), "conclusion": "pi"|"delta"}]
    """

    def __init__(self) -> None:
        self.train_db: Dict[str, dict] | None = None
        self.tot_pi: int = 0
        self.tot_delta: int = 0

    def fit(self, train_db: Dict[str, dict]) -> "MonotonicModel":
        self.train_db = train_db
        self.tot_pi = sum(1 for v in train_db.values() if v["conclusion"] == "pi")
        self.tot_delta = len(train_db) - self.tot_pi
        return self

    def predict(self, case: dict) -> Union[str, Tuple[str, float]]:
        if self.train_db is None:
            raise RuntimeError("Model must be fitted first.")
        cnt_le_pi, cnt_ge_delta = self._support_counts(case["set"])
        return self._predict_impl(cnt_le_pi, cnt_ge_delta, self.tot_pi, self.tot_delta)

    def _support_counts(self, X: set) -> Tuple[int, int]:
        """cnt_le_pi: # π-sets S⊆X ; cnt_ge_delta: # δ-sets S⊇X"""
        le_pi = ge_delta = 0
        for v in self.train_db.values():  # type: ignore
            S, dec = v["set"], v["conclusion"]
            if dec == "pi" and S.issubset(X):
                le_pi += 1
            if dec == "delta" and S.issuperset(X):
                ge_delta += 1
        return le_pi, ge_delta

    @abstractmethod
    def _predict_impl(self, cnt_le_pi: int, cnt_ge_delta: int, tot_pi: int, tot_delta: int):
        ...


# ================================================================
# 1) Strict Binary
# ================================================================
class MonotonicStrictBinary(MonotonicModel):
    """
    Decide only if exactly one side has any precedent:
      π if cnt_le_pi>0 and cnt_ge_delta=0
      δ if cnt_ge_delta>0 and cnt_le_pi=0
      abstain otherwise
    """
    def _predict_impl(self, cnt_le_pi, cnt_ge_delta, *_):
        if cnt_le_pi and not cnt_ge_delta:
            return "pi",    1.0
        if cnt_ge_delta and not cnt_le_pi:
            return "delta", 1.0
        return np.nan, np.nan


# ================================================================
# 2) Majority Binary (renamed to match desired API)
# ================================================================
class MonotonicMajorityBinary(MonotonicModel):
    """
    Decide by strict count majority among precedents; abstain on ties/no evidence.
    """
    def _predict_impl(self, cnt_le_pi, cnt_ge_delta, *_):
        if cnt_le_pi > cnt_ge_delta:
            return "pi",    1.0
        if cnt_ge_delta > cnt_le_pi:
            return "delta", 1.0
        return np.nan, np.nan


# ================================================================
# 3) BayesIndividualStrictMajority  (uses +1 smoothing)
#     – per-precedent Betas (α,β) = (r+1, s+1)
#     – MC majority on Bernoulli draws; confidence = side-aware P(S>0) or P(S<0)
# ================================================================
def _build_individual_betas_plus1(train_db: Dict[str, dict]) -> dict[str, tuple[float, float]]:
    """(α,β) = (r+1, s+1) — for the Individual model (pre log-odds)."""
    indiv: dict[str, tuple[float, float]] = {}
    for cid, info in train_db.items():
        S, conc = info["set"], info["conclusion"]
        if conc == "pi":
            r = sum(1.0 for v in train_db.values()
                    if v["conclusion"] == "pi"    and v["set"].issuperset(S))
            s = sum(1.0 for v in train_db.values()
                    if v["conclusion"] == "delta" and v["set"].issuperset(S))
        else:  # delta precedent
            r = sum(1.0 for v in train_db.values()
                    if v["conclusion"] == "delta" and v["set"].issubset(S))
            s = sum(1.0 for v in train_db.values()
                    if v["conclusion"] == "pi"    and v["set"].issubset(S))
        indiv[cid] = (r + 1.0, s + 1.0)
    return indiv


class MonotonicBayesIndividualMajority(MonotonicModel):
    """
    Unweighted per-precedent model with +1 smoothing.
    Predict by MC majority of 0/1 votes sampled from each influencing Beta.
    """
    def __init__(self, draws: int = 20_000, random_state: int | None = None):
        super().__init__()
        self.draws = int(draws)
        self.rng   = np.random.default_rng(random_state)

    def fit(self, train_db: Dict[str, dict]):
        super().fit(train_db)
        self._indiv_betas = _build_individual_betas_plus1(train_db)
        return self

    def predict(self, case: dict):
        self._current_set = case["set"]
        return super().predict(case)

    def _predict_impl(self, cnt_le_pi, cnt_ge_delta, *_):
        if (cnt_le_pi + cnt_ge_delta) == 0:
            return np.nan, np.nan

        # collect influencing Betas
        params_pi, params_del = [], []
        for cid, info in self.train_db.items():  # type: ignore
            if cid not in self._indiv_betas:
                continue
            S, conc = info["set"], info["conclusion"]
            if conc == "pi" and S.issubset(self._current_set):
                params_pi.append(self._indiv_betas[cid])
            elif conc == "delta" and S.issuperset(self._current_set):
                params_del.append(self._indiv_betas[cid])

        if not params_pi and not params_del:
            return np.nan, np.nan

        D, rng = self.draws, self.rng
        k_pi, k_del = len(params_pi), len(params_del)

        if k_pi:
            a_pi, b_pi = np.array(params_pi, float).T
            thetas_pi  = rng.beta(a_pi[:, None], b_pi[:, None], size=(k_pi, D))
            votes_pi   = rng.binomial(1, thetas_pi)
        else:
            votes_pi = np.zeros((1, D))

        if k_del:
            a_d, b_d = np.array(params_del, float).T
            thetas_d = rng.beta(a_d[:, None], b_d[:, None], size=(k_del, D))
            votes_del= rng.binomial(1, thetas_d)
        else:
            votes_del = np.zeros((1, D))

        S = votes_pi.sum(axis=0) - votes_del.sum(axis=0)
        mean_S = float(S.mean())
        if   mean_S > 0: return "pi",    float((S > 0).mean())
        elif mean_S < 0: return "delta", float((S < 0).mean())
        else:            return np.nan,  np.nan


# ================================================================
# 4) LogOddsNoClipPriorChainMC1
#     – per-precedent (r,s) with NO +1 (tiny ε guard)
#     – chain pruning: keep maximal π subsets / minimal δ supersets
#     – prior correction: (1−K)·log(π0/δ0), K=#kept precedents
#     – MC1 confidence: p* = E[σ(S)], decide π if p*>0.5 else δ
# ================================================================
EPS_BETA = 1e-9  # small guard to avoid 0 or negative Beta params

def _build_individual_betas_unweighted(train_db: Dict[str, dict]) -> dict[str, tuple[float, float]]:
    """NO +1 — used by log-odds model."""
    indiv: dict[str, tuple[float, float]] = {}
    for cid, info in train_db.items():
        S, conc = info["set"], info["conclusion"]
        if conc == "pi":
            r = sum(1.0 for v in train_db.values()
                    if v["conclusion"] == "pi"    and v["set"].issuperset(S))
            s = sum(1.0 for v in train_db.values()
                    if v["conclusion"] == "delta" and v["set"].issuperset(S))
        else:
            r = sum(1.0 for v in train_db.values()
                    if v["conclusion"] == "delta" and v["set"].issubset(S))
            s = sum(1.0 for v in train_db.values()
                    if v["conclusion"] == "pi"    and v["set"].issubset(S))
        a = max(float(r), EPS_BETA)
        b = max(float(s), EPS_BETA)
        indiv[cid] = (a, b)
    return indiv


def _influencer_params_chain_maxima(
    train_db: Dict[str, dict],
    indiv_betas: Dict[str, Tuple[float, float]],
    current_set: set
) -> tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    π side: keep only maximal subsets S ⊆ X (no S ⊂ S').
    δ side: keep only minimal supersets S ⊇ X (no S' ⊂ S).
    Returns lists of (alpha, beta).
    """
    pi_cands: List[tuple[str, set, tuple[float, float]]] = []
    de_cands: List[tuple[str, set, tuple[float, float]]] = []

    for cid, info in train_db.items():
        if cid not in indiv_betas:  # safety
            continue
        S, dec = info["set"], info["conclusion"]
        if dec == "pi" and S.issubset(current_set):
            pi_cands.append((cid, S, indiv_betas[cid]))
        elif dec == "delta" and S.issuperset(current_set):
            de_cands.append((cid, S, indiv_betas[cid]))

    # π: keep maximal by ⊆
    pi_kept: List[Tuple[float, float]] = []
    for i, (_, Si, par_i) in enumerate(pi_cands):
        keep = True
        for j, (_, Sj, _) in enumerate(pi_cands):
            if i != j and Si < Sj:  # strict subset ⇒ dominated
                keep = False
                break
        if keep:
            pi_kept.append(par_i)

    # δ: keep minimal by ⊇ (remove strict supersets)
    de_kept: List[Tuple[float, float]] = []
    for i, (_, Si, par_i) in enumerate(de_cands):
        keep = True
        for j, (_, Sj, _) in enumerate(de_cands):
            if i != j and Sj < Si:  # another candidate is strictly smaller
                keep = False
                break
        if keep:
            de_kept.append(par_i)

    return pi_kept, de_kept


def _logit_array(x, eps=1e-12):
    x = np.clip(x, eps, 1.0 - eps)
    return np.log(x) - np.log1p(-x)


class MonotonicBayesIndividualLogOddsNoClipPriorChainMC1(MonotonicModel):
    """
    Chain-pruned, unweighted, prior-corrected log-odds model with MC1 confidence.

      S(θ) = Σ logit(θ_π) − Σ logit(θ_δ) + (1−K)·log(π0/δ0)
      p*   = E_θ[ σ(S(θ)) ]  (MC estimate)

    Return ("pi", p*) if p*>0.5 else ("delta", 1−p*).
    """
    def __init__(self, draws: int = 20_000, random_state: int | None = None):
        super().__init__()
        self.draws = int(draws)
        self.rng   = np.random.default_rng(random_state)

    def fit(self, train_db: Dict[str, dict]):
        super().fit(train_db)
        # per-precedent (r,s) without +1
        self._indiv_betas = _build_individual_betas_unweighted(train_db)
        # prior odds from class frequencies (tiny guard)
        self._prior_log_odds = float(np.log((self.tot_pi + 1e-12) / (self.tot_delta + 1e-12)))
        return self

    def predict(self, case: dict):
        self._current_set = case["set"]
        return super().predict(case)

    def _predict_impl(self, *_):
        # chain-pruned influencing Betas
        params_pi, params_del = _influencer_params_chain_maxima(
            self.train_db, self._indiv_betas, self._current_set)  # type: ignore
        K = len(params_pi) + len(params_del)
        if K == 0:
            return np.nan, np.nan

        D, rng = self.draws, self.rng

        # sample and sum logits
        if params_pi:
            a, b = np.array(params_pi, float).T
            thetas = rng.beta(a[:, None], b[:, None], size=(len(params_pi), D))
            sum_pi = _logit_array(thetas).sum(axis=0)
        else:
            sum_pi = np.zeros(D, dtype=float)

        if params_del:
            a, b = np.array(params_del, float).T
            thetas = rng.beta(a[:, None], b[:, None], size=(len(params_del), D))
            sum_del = _logit_array(thetas).sum(axis=0)
        else:
            sum_del = np.zeros(D, dtype=float)

        prior_term = (1.0 - K) * self._prior_log_odds
        S = sum_pi - sum_del + prior_term
        p_star = float((1.0 / (1.0 + np.exp(-S))).mean())

        if p_star > 0.5:
            return "pi", p_star
        if p_star < 0.5:
            return "delta", 1.0 - p_star
        return np.nan, np.nan


# ---------------------------------------------------------------
# Backward-compat alias for older code (optional; not exported)
MonotonicStrictMajority = MonotonicMajorityBinary

__all__ = [
    "MonotonicStrictBinary",
    "MonotonicMajorityBinary",
    "MonotonicBayesIndividualMajority",
    "MonotonicBayesIndividualLogOddsNoClipPriorChainMC1",
]
