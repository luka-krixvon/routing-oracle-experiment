"""Routing-oracle paper: re-estimating the per-instance oracle under stochastic
decoding and decomposing the router-to-oracle gap.

Pure-math core (oracles, decompose, stats) is fully implemented and tested.
Generation/scoring expose interfaces (need model access).
"""
from . import oracles, decompose, stats  # noqa: F401
