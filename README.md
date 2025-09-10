# \# Bayesian Result Model (bayesrm)

# 

# This repository contains a compact set of \*\*decision models\*\* used in our experiments and figures.

# 

# The Python package `bayesrm/` exposes exactly four classes:

# 

# \- `MonotonicStrictBinary`

# \- `MonotonicMajorityBinary`

# \- `MonotonicBayesIndividualMajority`

# \- `MonotonicBayesIndividualLogOddsNoClipPriorChainMC1`

# 

# > \*\*Python ≥ 3.10\*\* (uses `X | Y` union types)  

# > \*\*Depends on:\*\* `numpy` (the notebook also uses `pandas`, `matplotlib`).

# 

# ---

# 

# \## 💡 Model name mapping (Paper ↔ Notebook ↔ Code)

# 

# The figures in the notebook use human-friendly labels that map to the names used in the \*\*paper\*\*.  

# \*\*In the paper, the fourth model is called simply “Naive-Bayes”.\*\*  

# Use the table below to translate between \*\*paper names\*\*, \*\*notebook labels\*\*, and the exact \*\*code classes\*\*.

# 

# | \*\*Paper name\*\*                                   | \*\*Notebook label\*\*                                     | \*\*Code class (import from `bayesrm`)\*\*                                  |

# |---|---|---|

# | \*\*Strict Binary\*\*                                | `Strict Binary`                                        | `MonotonicStrictBinary`                                                 |

# | \*\*Binary Majority\*\* \*(older drafts: “Strict Majority”)\* | `Binary Majority`                                | `MonotonicMajorityBinary`                                               |

# | \*\*Bayesian Individual Majority\*\*                 | `Bayesian Individual Majority`                         | `MonotonicBayesIndividualMajority`                                      |

# | \*\*Naive-Bayes\*\*                                  | `Naive Bayes (Chain+Prior, MC)` \*(or similar wording)\* | `MonotonicBayesIndividualLogOddsNoClipPriorChainMC1`                    |

# 

# \- Any notebook legend that reads \*\*“Naive Bayes (Chain+Prior, MC)”\*\* corresponds to the \*\*paper’s “Naive-Bayes”\*\* and maps to the class `MonotonicBayesIndividualLogOddsNoClipPriorChainMC1`.

# 

# ---

# 

# \## What each model does (short version)

# 

# \- \*\*MonotonicStrictBinary\*\* – Decide only if exactly one side has precedent:  

# &nbsp; π if (∃ π precedents S ⊆ X) and (no δ supersets S ⊇ X); δ symmetrically; otherwise \*\*abstain\*\*.

# 

# \- \*\*MonotonicMajorityBinary\*\* – \*\*Strict count majority\*\* between π-subsets and δ-supersets; ties/no evidence → \*\*abstain\*\*.

# 

# \- \*\*MonotonicBayesIndividualMajority\*\* – Per-precedent Beta(r+1, s+1) with +1 smoothing; sample Bernoulli votes via MC; majority wins; confidence is side-aware P(S>0) or P(S<0).

# 

# \- \*\*MonotonicBayesIndividualLogOddsNoClipPriorChainMC1\*\* (\*paper: \*\*Naive-Bayes\*\*\*) – Per-precedent \*\*unweighted\*\* Betas (no +1; ε guard), \*\*chain-pruned\*\* (keep maximal π subsets / minimal δ supersets), \*\*prior-corrected\*\* log-odds; confidence via MC estimate \\(p^\\\*=\\mathbb{E}\[\\sigma(S)]\\).

# 

# ---

# 

# \## Install / Import

# 

# ```python

# from bayesrm import (

# &nbsp;   MonotonicStrictBinary,

# &nbsp;   MonotonicMajorityBinary,

# &nbsp;   MonotonicBayesIndividualMajority,

# &nbsp;   MonotonicBayesIndividualLogOddsNoClipPriorChainMC1,

# )





