# BEAM: Budget-Efficient Agent Management

<p align="center">
  <b>Token-efficient multi-agent inference optimization toolkit</b>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#strategies">Strategies</a> •
  <a href="#api-reference">API Reference</a> •
  <a href="#examples">Examples</a>
</p>

---

## Overview

BEAM is a toolkit designed to **reduce token consumption** in multi-agent LLM systems while maintaining output quality. It addresses the challenge of high inference costs in multi-agent architectures by learning which agent connections and communications are essential.

### Key Features

- **Three Optimization Strategies**: AgentPrune, AgentDropout, AgentBayesian
- **Extensible Prompt Management**: Domain-specific prompt templates with registry
- **Framework Integration**: Lightweight utilities for LangChain and LangGraph
- **Flexible Agent Design**: Support custom execution functions or LLM-based agents
- **Graph-based Architecture**: Spatial (same-round) and temporal (cross-round) connections

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Agent System                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │ Agent 1 │───▶│ Agent 2 │───▶│ Agent 3 │───▶│Decision │      │
│  └─────────┘    └─────────┘    └─────────┘    └─────────┘      │
│       │              │              │                           │
│       └──────────────┴──────────────┘                           │
│                      │                                          │
│              BEAM Optimization                                  │
│                      ▼                                          │
│  ┌─────────┐              ┌─────────┐    ┌─────────┐           │
│  │ Agent 1 │─────────────▶│ Agent 3 │───▶│Decision │           │
│  └─────────┘              └─────────┘    └─────────┘           │
│                                                                 │
│  Result: Fewer tokens, maintained accuracy                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Basic Installation

```bash
git clone https://github.com/erwinmsmith/BEAM.git
cd BEAM
pip install -e .
```

### With Optional Dependencies

```bash
# LangChain integration
pip install -e ".[langchain]"

# LangGraph integration  
pip install -e ".[langgraph]"

# Bayesian/MCMC support
pip install -e ".[bayesian]"

# All dependencies
pip install -e ".[all]"

# Development tools
pip install -e ".[dev]"
```

### Requirements

- Python >= 3.9
- PyTorch >= 2.0.0
- OpenAI >= 1.0.0 (for LLM support)

---

## Quick Start

### 1. Configure Your System

```python
from beam import BEAMConfig, AgentConfig, OptimizationConfig

config = BEAMConfig(
    # Define your agents
    agents=[
        AgentConfig(name="Analyzer", count=2),
        AgentConfig(name="Solver", count=3),
        AgentConfig(name="Verifier", count=1),
    ],
    
    # Multi-round reasoning
    num_rounds=2,
    
    # Optimization settings
    optimization=OptimizationConfig(
        strategy="prune",           # "prune", "dropout", or "bayesian"
        optimize_spatial=True,      # Optimize same-round connections
        optimize_temporal=True,     # Optimize cross-round connections
        pruning_rate=0.25,          # Target 25% edge reduction
        learning_rate=0.01,
    ),
    
    # Decision aggregation
    decision_method="reference",    # "reference", "majority", or "direct"
    domain="math"
)
```

### 2. Create Agents

#### Option A: Using Custom Functions

```python
from beam import create_agent_node

def analyze_problem(state):
    """Custom analysis logic."""
    task = state["task"]
    context = state.get("context", "")
    # Your analysis logic here
    return f"Analysis: The problem requires solving {task}"

def solve_problem(state):
    """Custom solving logic."""
    task = state["task"]
    analysis = state.get("context", "")
    # Your solving logic here
    return f"Solution based on analysis: {analysis}"

# Create nodes
analyzer = create_agent_node(role="Analyzer", execute_fn=analyze_problem)
solver = create_agent_node(role="Solver", execute_fn=solve_problem)
```

#### Option B: Using LLM with Prompts

```python
from beam import create_agent_node, LLMRegistry

# Get LLM instance
llm = LLMRegistry.get("gpt-4o")

# Create agent with prompts
analyzer = create_agent_node(
    role="Analyzer",
    llm=llm,
    system_prompt="""You are a problem analyzer. Your task is to:
1. Identify key information in the problem
2. Determine what type of problem this is
3. Outline the approach to solve it""",
    user_prompt_template="""Problem: {task}

Previous analysis (if any):
{context}

Provide your analysis:"""
)

solver = create_agent_node(
    role="Solver", 
    llm=llm,
    system_prompt="""You are a problem solver. Based on the analysis provided,
solve the problem step by step and provide a clear final answer.""",
    user_prompt_template="""Problem: {task}

Analysis from other agents:
{context}

Your solution:"""
)
```

### 3. Use Prompt Templates (Recommended)

```python
from beam import PromptSet, PromptRegistry

# Create a reusable prompt set
math_prompts = PromptSet(name="math", description="Math problem solving")

# Add role-specific prompts
math_prompts.add_role(
    role="analyzer",
    system="""You are a mathematical analyst. Break down problems into components:
- Identify given information
- Identify what needs to be found
- Suggest solution strategies""",
    user="Problem: {task}\n\nContext: {context}\n\nYour analysis:"
)

math_prompts.add_role(
    role="solver",
    system="""You are a math solver. Show your work step by step.
Be precise with calculations and clearly state your final answer.""",
    user="Problem: {task}\n\nAnalysis: {context}\n\nSolution:"
)

math_prompts.add_role(
    role="verifier",
    system="""You verify mathematical solutions. Check for:
- Calculation errors
- Logic errors
- Missing steps
Confirm or correct the answer.""",
    user="Problem: {task}\n\nSolution to verify: {context}\n\nVerification:"
)

# Set decision template
math_prompts.set_decision_template(
    system="Synthesize multiple solutions and provide the final answer.",
    user="Problem: {task}\n\nSolutions:\n{context}\n\nFinal answer:"
)

# Register for global access
PromptRegistry.register("math", math_prompts)

# Use anywhere in your code
prompts = PromptRegistry.get("math")
system, user = prompts.get_prompt("solver", task="2+2=?", context="Simple addition")
```

### 4. Train and Run Optimization

```python
from beam import AgentPrune, AgentGraph

# Create strategy
strategy = AgentPrune(config)

# Build graph (connects agents based on config)
graph = AgentGraph(config)
graph.add_nodes([analyzer, solver, verifier])
strategy.set_graph(graph)

# Prepare training data
train_data = [
    {"task": "What is 15% of 80?", "answer": "12"},
    {"task": "Solve: 2x + 5 = 13", "answer": "4"},
    {"task": "Calculate 3^4", "answer": "81"},
    # ... more examples
]

# Define evaluation function
def eval_fn(prediction: str, answer: str) -> float:
    """Return 1.0 for correct, 0.0 for incorrect."""
    try:
        # Extract numbers and compare
        pred_nums = [float(s) for s in prediction.split() if s.replace('.','').isdigit()]
        ans_num = float(answer)
        return 1.0 if ans_num in pred_nums else 0.0
    except:
        return 0.0

# Train (learns which edges to prune)
training_stats = strategy.train(
    train_data=train_data,
    eval_fn=eval_fn,
    epochs=10,
    batch_size=4
)

print(f"Training accuracy: {training_stats['final_accuracy']:.2%}")
print(f"Edges pruned: {training_stats['edges_pruned']}")

# Run optimized inference
result, metadata = await strategy.run({"task": "What is 25% of 200?"})
print(f"Answer: {result}")
print(f"Tokens used: {metadata['tokens_used']}")
print(f"Agents activated: {metadata['active_agents']}")
```

---

## Strategies

### AgentPrune

Learns edge importance through policy gradient optimization and prunes low-importance connections.

```python
from beam import AgentPrune, OptimizationConfig

config = BEAMConfig(
    # ...
    optimization=OptimizationConfig(
        strategy="prune",
        optimize_spatial=True,
        optimize_temporal=True,
        pruning_rate=0.3,        # Prune 30% of edges
        initial_probability=0.5, # Initial edge probability
    )
)

strategy = AgentPrune(config)
```

**Best for**: Dense agent networks where many connections are redundant.

**How it works**:
1. Initializes learnable logits for each potential edge
2. Uses Gumbel-Softmax for differentiable edge sampling
3. Optimizes via policy gradient based on task performance
4. Prunes edges below learned threshold

### AgentDropout

Learns which agents can be skipped entirely during inference.

```python
from beam import AgentDropout, OptimizationConfig

config = BEAMConfig(
    # ...
    optimization=OptimizationConfig(
        strategy="dropout",
        optimize_spatial=True,
        dropout_rate=0.2,        # Target 20% agent skip rate
    )
)

strategy = AgentDropout(config)
```

**Best for**: Systems with redundant agents where some can be skipped without quality loss.

**How it works**:
1. Learns skip probability for each agent
2. During training, samples skip decisions
3. Agents with high skip probability are dropped during inference
4. Maintains ensemble diversity while reducing computation

### AgentBayesian

Uses Bayesian optimization with optional MCMC sampling for uncertainty-aware pruning.

```python
from beam import AgentBayesian, OptimizationConfig

config = BEAMConfig(
    # ...
    optimization=OptimizationConfig(
        strategy="bayesian",
        optimize_spatial=True,
        optimize_temporal=True,
        use_mcmc=True,           # Enable MCMC sampling
        mcmc_samples=100,        # Number of MCMC samples
        prior_mean=0.5,          # Prior edge probability
    )
)

strategy = AgentBayesian(config)
```

**Best for**: When you need confidence estimates or have limited training data.

**How it works**:
1. Maintains posterior distribution over edge weights
2. Uses MCMC (optional) to sample from posterior
3. Provides uncertainty estimates for pruning decisions
4. More robust with small datasets

---

## API Reference

### Core Classes

#### `BEAMConfig`

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

#### `AgentNode`

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

#### `AgentGraph`

Manages agent connections and execution flow.

```python
graph = AgentGraph(config)
graph.add_node(node)
graph.add_nodes([node1, node2, node3])

# Run inference
results = await graph.run({"task": "..."}, num_rounds=2)
```

#### `PromptSet`

Collection of prompts for a domain.

```python
prompts = PromptSet(name="domain_name")
prompts.add_role(role, system, user)
prompts.set_decision_template(system, user)
prompts.get_prompt(role, **variables)
prompts.save("prompts.json")
prompts.load("prompts.json")
```

#### `PromptRegistry`

Global registry for prompt sets.

```python
PromptRegistry.register(name, prompt_set)
PromptRegistry.get(name)
PromptRegistry.keys()
PromptRegistry.load_from_file(name, path)
```

### LLM Classes

#### `BaseLLM`

Abstract base class for LLM implementations.

```python
class BaseLLM(ABC):
    @abstractmethod
    def gen(self, messages: List[Dict]) -> str: ...
    
    @abstractmethod
    async def agen(self, messages: List[Dict]) -> str: ...
```

#### `LLMRegistry`

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

### LangChain

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

### LangGraph

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

## Examples

### Complete Math Solving Example

```python
import asyncio
from beam import (
    BEAMConfig, AgentConfig, OptimizationConfig,
    AgentPrune, AgentGraph, create_agent_node,
    PromptSet, PromptRegistry, LLMRegistry
)

async def main():
    # 1. Setup prompts
    prompts = PromptSet(name="math")
    prompts.add_role("solver", 
        system="Solve math problems step by step.",
        user="Problem: {task}\nContext: {context}\nSolution:")
    prompts.add_role("verifier",
        system="Verify mathematical solutions.",
        user="Problem: {task}\nSolution: {context}\nVerification:")
    PromptRegistry.register("math", prompts)
    
    # 2. Configure system
    config = BEAMConfig(
        agents=[
            AgentConfig(name="solver", count=3),
            AgentConfig(name="verifier", count=1),
        ],
        num_rounds=1,
        optimization=OptimizationConfig(
            strategy="prune",
            optimize_spatial=True,
            pruning_rate=0.25,
        )
    )
    
    # 3. Create agents
    llm = LLMRegistry.get("gpt-4o")
    math_prompts = PromptRegistry.get("math")
    
    agents = []
    for i in range(3):
        sys, usr = math_prompts.get_prompt("solver", task="{task}", context="{context}")
        agents.append(create_agent_node(
            role=f"Solver_{i}",
            llm=llm,
            system_prompt=sys,
            user_prompt_template=usr
        ))
    
    sys, usr = math_prompts.get_prompt("verifier", task="{task}", context="{context}")
    agents.append(create_agent_node(
        role="Verifier",
        llm=llm,
        system_prompt=sys,
        user_prompt_template=usr
    ))
    
    # 4. Build and train
    strategy = AgentPrune(config)
    graph = AgentGraph(config)
    graph.add_nodes(agents)
    strategy.set_graph(graph)
    
    train_data = [
        {"task": "2 + 2", "answer": "4"},
        {"task": "10 * 5", "answer": "50"},
        {"task": "100 / 4", "answer": "25"},
    ]
    
    def eval_fn(pred, ans):
        return 1.0 if ans in pred else 0.0
    
    stats = strategy.train(train_data, eval_fn, epochs=5)
    
    # 5. Run inference
    result, meta = await strategy.run({"task": "What is 15% of 200?"})
    print(f"Result: {result}")
    print(f"Tokens saved: {meta.get('tokens_saved', 'N/A')}")

if __name__ == "__main__":
    asyncio.run(main())
```

See `beam/examples/` for more examples.

---

## Project Structure

```
BEAM/
├── beam/
│   ├── __init__.py          # Package exports
│   ├── core/
│   │   ├── config.py        # BEAMConfig, OptimizationConfig, AgentConfig
│   │   ├── graph.py         # AgentGraph - manages agent connections
│   │   ├── node.py          # AgentNode - base agent class
│   │   ├── llm.py           # BaseLLM, LLMRegistry
│   │   └── optimizer.py     # TokenOptimizer - training orchestration
│   ├── strategies/
│   │   ├── prune.py         # AgentPrune strategy
│   │   ├── dropout.py       # AgentDropout strategy
│   │   └── bayesian.py      # AgentBayesian strategy
│   ├── prompts/
│   │   ├── template.py      # PromptTemplate, PromptSet
│   │   └── registry.py      # PromptRegistry
│   ├── integrations/
│   │   ├── langchain.py     # LangChain utilities
│   │   └── langgraph.py     # LangGraph utilities
│   └── examples/
│       ├── basic_usage.py
│       └── prompts_example.py
├── pyproject.toml           # Package configuration
├── README.md
└── requirements.txt
```

---

## Branches

| Branch | Description |
|--------|-------------|
| `main` | Production-ready engineered package |
| `experiment-version` | Original experimental code (preserved for reference) |

---

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

## License

MIT License

---

## Citation

If you use BEAM in your research, please cite:

```bibtex
@software{beam2024,
  title={BEAM: Budget-Efficient Agent Management},
  author={BEAM Team},
  year={2024},
  url={https://github.com/erwinmsmith/BEAM}
}
```

