# BEAM: Budget-Efficient Agent Management

A toolkit for **token-efficient multi-agent inference optimization**.

BEAM provides three optimization strategies to reduce token consumption in multi-agent LLM systems while maintaining output quality:

- **AgentPrune**: Learn edge importance and prune low-weight connections
- **AgentDropout**: Dynamically skip agents during inference
- **AgentBayesian**: Bayesian optimization with optional MCMC sampling

## Installation

```bash
pip install -e .

# With optional dependencies
pip install -e ".[langchain]"      # LangChain integration
pip install -e ".[langgraph]"      # LangGraph integration
pip install -e ".[bayesian]"       # Bayesian/MCMC support
pip install -e ".[all]"            # Everything
```

## Quick Start

### 1. Basic Configuration

```python
from beam import BEAMConfig, AgentConfig, OptimizationConfig

config = BEAMConfig(
    agents=[
        AgentConfig(name="Solver", count=4),
        AgentConfig(name="Verifier", count=1),
    ],
    num_rounds=2,
    optimization=OptimizationConfig(
        strategy="prune",
        optimize_spatial=True,
        optimize_temporal=True,
        pruning_rate=0.25,
    )
)
```

### 2. Create Agents

```python
from beam import create_agent_node, LLMRegistry

# With custom execution function
def solver_fn(state):
    task = state["task"]
    context = state["context"]
    return f"Solution: {task}"

solver = create_agent_node(
    role="Solver",
    execute_fn=solver_fn
)

# With LLM and prompts
llm = LLMRegistry.get("gpt-4o", api_key="...")
solver = create_agent_node(
    role="Solver",
    llm=llm,
    system_prompt="You are a problem solver.",
    user_prompt_template="Solve: {task}\nContext: {context}"
)
```

### 3. Use Optimization Strategies

```python
from beam import AgentPrune

# Create strategy
prune = AgentPrune(config)
prune.set_graph(your_graph)

# Train
train_data = [{"task": "2+2", "answer": "4"}, ...]
stats = prune.train(train_data, eval_fn)

# Run optimized inference
result, metadata = await prune.run({"task": "What is 10/2?"})
```

### 4. Prompt Template Management

```python
from beam import PromptSet, PromptRegistry

# Create prompt set
prompts = PromptSet(name="math")
prompts.add_role(
    "solver",
    system="You are a math expert.",
    user="Problem: {task}\nContext: {context}"
)
prompts.set_decision_template(
    system="Synthesize solutions.",
    user="Solutions: {context}\nFinal answer:"
)

# Register for reuse
PromptRegistry.register("math", prompts)

# Use later
math_prompts = PromptRegistry.get("math")
system, user = math_prompts.get_prompt("solver", task="2+2", context="")
```

## Integration with LangChain/LangGraph

BEAM provides lightweight utilities to integrate with existing frameworks:

```python
from beam.integrations.langchain import LangChainLLMWrapper
from beam.integrations.langgraph import create_beam_node, BEAMState

# Wrap LangChain LLM for BEAM
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
beam_llm = LangChainLLMWrapper(llm)

# Use with BEAM agents
solver = create_agent_node(role="Solver", llm=beam_llm, ...)
```

## Project Structure

```
beam/
├── core/           # Core components
│   ├── config.py   # Configuration classes
│   ├── graph.py    # Agent graph implementation
│   ├── node.py     # Agent node base class
│   ├── llm.py      # LLM abstraction layer
│   └── optimizer.py # Token optimizer
├── strategies/     # Optimization strategies
│   ├── prune.py    # AgentPrune
│   ├── dropout.py  # AgentDropout
│   └── bayesian.py # AgentBayesian
├── prompts/        # Prompt template system
│   ├── template.py # PromptTemplate, PromptSet
│   └── registry.py # PromptRegistry
├── integrations/   # Framework integrations
│   ├── langchain.py
│   └── langgraph.py
└── examples/       # Usage examples
```

## Strategies Overview

| Strategy | Description | Best For |
|----------|-------------|----------|
| **AgentPrune** | Learns edge importance, prunes low-weight connections | Dense agent networks |
| **AgentDropout** | Learns which agents to skip per round | Redundant configurations |
| **AgentBayesian** | Bayesian optimization with uncertainty estimates | Confidence-aware pruning |

## Branches

- `main`: Production-ready engineered package
- `experiment-version`: Original experimental code (preserved for reference)

