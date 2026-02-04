"""
LangChain integration utilities for BEAM toolkit.

This module provides lightweight adapters and utilities to integrate
BEAM's token optimization strategies with LangChain workflows.

BEAM does not replace LangChain - it extends it with token-efficient
multi-agent optimization capabilities.
"""

from typing import Dict, Any, List, Optional, Callable, Union
from abc import ABC, abstractmethod


class LangChainLLMWrapper:
    """
    Wrapper to use LangChain LLMs with BEAM agents.
    
    Example:
        ```python
        from langchain_openai import ChatOpenAI
        from beam.integrations.langchain import LangChainLLMWrapper
        
        llm = ChatOpenAI(model="gpt-4")
        beam_llm = LangChainLLMWrapper(llm)
        
        # Use with BEAM agents
        agent = SolverAgent(llm=beam_llm)
        ```
    """

    def __init__(self, langchain_llm):
        """
        Args:
            langchain_llm: A LangChain BaseChatModel or LLM instance
        """
        self.llm = langchain_llm

    async def agen(self, messages: List[Dict], **kwargs) -> str:
        """Generate response asynchronously."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        except ImportError:
            raise ImportError("langchain-core required: pip install langchain-core")
        
        lc_messages = self._convert_messages(messages)
        response = await self.llm.ainvoke(lc_messages)
        return response.content

    def gen(self, messages: List[Dict], **kwargs) -> str:
        """Generate response synchronously."""
        try:
            from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        except ImportError:
            raise ImportError("langchain-core required: pip install langchain-core")
        
        lc_messages = self._convert_messages(messages)
        response = self.llm.invoke(lc_messages)
        return response.content

    def _convert_messages(self, messages: List[Dict]) -> List:
        """Convert dict messages to LangChain message format."""
        from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
        
        lc_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "user":
                lc_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
        
        return lc_messages


class LangChainCallbackHandler:
    """
    Callback handler to track token usage in LangChain workflows.
    
    Use this to monitor token consumption and integrate with BEAM's
    optimization metrics.
    
    Example:
        ```python
        from beam.integrations.langchain import LangChainCallbackHandler
        
        handler = LangChainCallbackHandler()
        llm = ChatOpenAI(callbacks=[handler])
        
        # After running
        print(f"Tokens used: {handler.total_tokens}")
        ```
    """

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def on_llm_end(self, response, **kwargs):
        """Called when LLM finishes."""
        self.call_count += 1
        if hasattr(response, 'llm_output') and response.llm_output:
            usage = response.llm_output.get('token_usage', {})
            self.prompt_tokens += usage.get('prompt_tokens', 0)
            self.completion_tokens += usage.get('completion_tokens', 0)
            self.total_tokens += usage.get('total_tokens', 0)

    def reset(self):
        """Reset counters."""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.call_count = 0

    def get_stats(self) -> Dict[str, int]:
        """Get token usage statistics."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "call_count": self.call_count
        }


def wrap_langchain_runnable(runnable, role: str = "Assistant") -> Callable:
    """
    Wrap a LangChain Runnable as a BEAM-compatible node function.
    
    Args:
        runnable: LangChain Runnable (chain, agent, etc.)
        role: Role description for the agent
        
    Returns:
        A callable that can be used with BEAM's AgentNode
        
    Example:
        ```python
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        prompt = ChatPromptTemplate.from_messages([...])
        chain = prompt | ChatOpenAI()
        
        # Wrap for BEAM
        node_fn = wrap_langchain_runnable(chain, role="Analyst")
        ```
    """
    async def wrapped_fn(inputs: Dict[str, Any]) -> str:
        if hasattr(runnable, 'ainvoke'):
            result = await runnable.ainvoke(inputs)
        else:
            import asyncio
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: runnable.invoke(inputs)
            )
        
        # Extract content
        if hasattr(result, 'content'):
            return result.content
        elif isinstance(result, dict) and 'output' in result:
            return result['output']
        elif isinstance(result, str):
            return result
        else:
            return str(result)
    
    wrapped_fn.__beam_role__ = role
    return wrapped_fn


def create_beam_chain(
    runnables: List,
    roles: Optional[List[str]] = None,
    optimization_strategy: str = "prune"
) -> Dict[str, Any]:
    """
    Create a BEAM-optimized chain configuration from LangChain runnables.
    
    This returns a configuration dict that can be used to initialize
    BEAM's optimization strategies.
    
    Args:
        runnables: List of LangChain Runnables
        roles: Optional role names for each runnable
        optimization_strategy: 'prune', 'dropout', or 'bayesian'
        
    Returns:
        Configuration dict for BEAM
        
    Example:
        ```python
        from beam.integrations.langchain import create_beam_chain
        from beam import BEAMConfig, AgentPrune
        
        chain_config = create_beam_chain(
            [researcher_chain, analyst_chain, writer_chain],
            roles=["Researcher", "Analyst", "Writer"],
            optimization_strategy="prune"
        )
        
        # Use with BEAM
        config = BEAMConfig.from_dict(chain_config)
        ```
    """
    roles = roles or [f"Agent_{i}" for i in range(len(runnables))]
    
    return {
        "agents": [
            {"name": role, "count": 1}
            for role in roles
        ],
        "optimization": {
            "strategy": optimization_strategy,
            "optimize_spatial": True,
            "optimize_temporal": True
        },
        "_langchain_runnables": runnables,
        "_langchain_roles": roles
    }
