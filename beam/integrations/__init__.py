"""
Integration utilities for BEAM toolkit.

Provides lightweight adapters to integrate BEAM with:
- LangChain: LLM wrappers, callback handlers, runnable wrappers
- LangGraph: State types, node wrappers, conditional edges
"""

from beam.integrations.langchain import (
    LangChainLLMWrapper,
    LangChainCallbackHandler,
    wrap_langchain_runnable,
    create_beam_chain,
)
from beam.integrations.langgraph import (
    BEAMState,
    create_beam_node,
    create_skip_condition,
    apply_beam_masks,
    get_langgraph_config,
)

__all__ = [
    # LangChain
    "LangChainLLMWrapper",
    "LangChainCallbackHandler",
    "wrap_langchain_runnable",
    "create_beam_chain",
    # LangGraph
    "BEAMState",
    "create_beam_node",
    "create_skip_condition",
    "apply_beam_masks",
    "get_langgraph_config",
]
