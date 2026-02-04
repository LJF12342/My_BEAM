"""Optimization strategies for BEAM toolkit."""

from beam.strategies.prune import AgentPrune
from beam.strategies.dropout import AgentDropout
from beam.strategies.bayesian import AgentBayesian

__all__ = [
    "AgentPrune",
    "AgentDropout", 
    "AgentBayesian",
]
