# API Reference

The BEAM API Reference provides a detailed technical overview of the framework's components. It is organized into three key areas: Core Classes for building and managing agent graphs, LLM Classes for model interfacing and registration, and Integrations for bridging BEAM with external ecosystems like LangChain and LangGraph.
---

## Core Classes

**BEAMConfig:**

Main configuration class for BEAM systems.
```python
@dataclass
class BEAMConfig:
    agents: List[AgentConfig]           # Agent definitions
    num_rounds: int = 1                 # Number of reasoning rounds
    optimization: OptimizationConfig    # Optimization settings
    decision_method: str = "reference"  # "reference", "majority", "direct"
    domain: str = ""                    # Task domain
```
**AgentNode:**

Base class for all agents in the graph.

```python
node = AgentNode(
    id="unique_id",                     # Optional, auto-generated if not provided
    agent_name="Solver",                # Agent type name
    role="Problem Solver",              # Role description
    domain="math",                      # Task domain
    llm=llm_instance,                   # LLM for generation (optional)
    execute_fn=custom_function,         # Custom execution (optional)
    system_prompt="...",                # System prompt for LLM
    user_prompt_template="...",         # User prompt template
)

# Execute
result = node.execute({"task": "..."})
result = await node.async_execute({"task": "..."})
```
**AgentGraph:**

Manages agent connections and execution flow.


```python
graph = AgentGraph(config)
graph.add_node(node)
graph.add_nodes([node1, node2, node3])

# Run inference
results = await graph.run({"task": "..."}, num_rounds=2)
```
**PromptSet:**

Collection of prompts for a domain.


```python
prompts = PromptSet(name="domain_name")
prompts.add_role(role, system, user)
prompts.set_decision_template(system, user)
prompts.get_prompt(role, **variables)
prompts.save("prompts.json")
prompts.load("prompts.json")
```
**PromptRegistry:**

Global registry for prompt sets.
```python
PromptRegistry.register(name, prompt_set)
PromptRegistry.get(name)
PromptRegistry.keys()
PromptRegistry.load_from_file(name, path)
```
---

## LLM Classes
**BaseLLM:**

Abstract base class for LLM implementations.
```python
class BaseLLM(ABC):
    @abstractmethod
    def gen(self, messages: List[Dict]) -> str: ...
    
    @abstractmethod
    async def agen(self, messages: List[Dict]) -> str: ...
```
**LLMRegistry:**

Registry for LLM implementations.
```python
# Get LLM instance
llm = LLMRegistry.get("gpt-4o")
llm = LLMRegistry.get("deepseek-chat")

# Register custom LLM
@LLMRegistry.register("custom")
class CustomLLM(BaseLLM):
    ...
```
---

## Integration
**LangChain:**

```python
from beam.integrations.langchain import (
    LangChainLLMWrapper,
    LangChainCallbackHandler,
    wrap_langchain_runnable,
)

# Wrap LangChain LLM for use with BEAM
from langchain_openai import ChatOpenAI
langchain_llm = ChatOpenAI(model="gpt-4")
beam_llm = LangChainLLMWrapper(langchain_llm)

# Use with BEAM agents
agent = create_agent_node(role="Solver", llm=beam_llm, ...)

# Track token usage with callback
callback = LangChainCallbackHandler()
# Use callback in your LangChain chains

# Wrap existing runnable
from langchain_core.runnables import RunnableSequence
wrapped = wrap_langchain_runnable(your_chain, beam_config)
```
**LangGraph:**

Registry for LLM implementations.
```python
from beam.integrations.langgraph import (
    BEAMState,
    create_beam_node,
    create_skip_condition,
    apply_beam_masks,
)

# Use BEAM state in your graph
from langgraph.graph import StateGraph

class MyState(BEAMState):
    custom_field: str

# Create BEAM-aware node
@create_beam_node(agent_id="solver")
def solver_node(state: MyState) -> dict:
    # Your logic here
    return {"result": "..."}

# Create skip condition based on BEAM masks
skip_condition = create_skip_condition("solver", beam_strategy)

# Build graph
graph = StateGraph(MyState)
graph.add_node("solver", solver_node)
graph.add_conditional_edges("start", skip_condition, {...})
```
---