"""ML meta-model: learns which rule-generated setups actually win.

This package NEVER generates signals on its own. The rule engine proposes;
the meta-model (trained on the backtester's simulated trades) re-scores the
proposal, and the two probabilities are blended in log-odds space by
winrate.blend_win_rate — always inside the honest 30–78% cap, and only once
the model has earned trust (enough training trades, positive OOS AUC).
"""
from .predict import MetaModel  # noqa: F401
