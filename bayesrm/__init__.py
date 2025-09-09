from importlib import import_module

_m = import_module(".all_models", __package__)

# Prefer canonical names; fall back to short aliases if present.
MonotonicStrictBinary = getattr(_m, "MonotonicStrictBinary",
                                getattr(_m, "StrictBinary", None))
MonotonicMajorityBinary = getattr(_m, "MonotonicMajorityBinary",
                                  getattr(_m, "StrictMajority", None))
MonotonicBayesIndividualMajority = getattr(
    _m, "MonotonicBayesIndividualMajority",
    getattr(_m, "BayesIndividualStrictMajority", None)
)
MonotonicBayesIndividualLogOddsNoClipPriorChainMC1 = getattr(
    _m, "MonotonicBayesIndividualLogOddsNoClipPriorChainMC1",
    getattr(_m, "LogOddsNoClipPriorChainMC1", None)
)

__all__ = [
    "MonotonicStrictBinary",
    "MonotonicMajorityBinary",
    "MonotonicBayesIndividualMajority",
    "MonotonicBayesIndividualLogOddsNoClipPriorChainMC1",
]
