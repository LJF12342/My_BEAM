"""
Example: Basic BEAM usage.

This example demonstrates the core workflow of using BEAM
for token-efficient multi-agent inference.
"""

import asyncio
from beam import (
    BEAMConfig,
    AgentConfig,
    OptimizationConfig,
    AgentGraph,
    AgentNode,
    create_agent_node,
    AgentPrune,
    AgentDropout,
    PromptSet,
)


# Example 1: Basic configuration
def create_basic_config():
    """Create a basic BEAM configuration."""
    
    config = BEAMConfig(
        # Define agents
        agents=[
            AgentConfig(name="Solver", count=3),
            AgentConfig(name="Verifier", count=1),
        ],
        
        # Graph settings
        num_rounds=2,
        
        # LLM settings (configure with your API)
        # llm=LLMConfig(
        #     model_name="gpt-4o",
        #     api_key="your-api-key"
        # ),
        
        # Optimization settings
        optimization=OptimizationConfig(
            strategy="prune",
            optimize_spatial=True,
            optimize_temporal=True,
            pruning_rate=0.25,
        ),
        
        domain="math"
    )
    
    return config


# Example 2: Create agents with custom execution functions
def create_agents_with_functions():
    """Create agents using custom execution functions."""
    
    # Define custom execution functions
    def solver_fn(state):
        """Custom solver logic."""
        task = state["task"]
        context = state.get("context", "")
        # Your solving logic here
        return f"Solution for: {task}"
    
    def verifier_fn(state):
        """Custom verifier logic."""
        task = state["task"]
        context = state.get("context", "")
        # Your verification logic here
        return f"Verified: {context}"
    
    # Create nodes
    solver = create_agent_node(
        role="Solver",
        execute_fn=solver_fn
    )
    
    verifier = create_agent_node(
        role="Verifier",
        execute_fn=verifier_fn
    )
    
    return [solver, verifier]


# Example 3: Create agents with LLM and prompts
def create_agents_with_llm(llm):
    """Create agents using LLM with prompt templates."""
    
    solver = create_agent_node(
        role="Solver",
        llm=llm,
        system_prompt="""You are a problem solver. Solve the given problem step by step.
Show your reasoning and provide a clear final answer.""",
        user_prompt_template="""Problem: {task}

Other solutions:
{context}

Your solution:"""
    )
    
    verifier = create_agent_node(
        role="Verifier",
        llm=llm,
        system_prompt="""You verify solutions for correctness.
Check the work and confirm or correct the answer.""",
        user_prompt_template="""Problem: {task}

Solutions to verify:
{context}

Your verification:"""
    )
    
    return [solver, verifier]


# Example 4: Using the AgentPrune strategy
async def prune_example():
    """Example of using AgentPrune for edge optimization."""
    
    config = create_basic_config()
    
    # Create the pruning strategy
    prune = AgentPrune(config)
    
    # You would set up your graph here
    # prune.set_graph(your_graph)
    
    # Training data format
    train_data = [
        {"task": "What is 2 + 2?", "answer": "4"},
        {"task": "What is 3 * 5?", "answer": "15"},
    ]
    
    # Evaluation function
    def eval_fn(prediction, answer):
        # Extract number from prediction and compare
        try:
            pred_num = float(''.join(c for c in prediction if c.isdigit() or c == '.'))
            ans_num = float(answer)
            return 1.0 if pred_num == ans_num else 0.0
        except:
            return 0.0
    
    # Train (requires graph to be set)
    # stats = prune.train(train_data, eval_fn)
    
    # Run inference
    # result, metadata = await prune.run({"task": "What is 10 / 2?"})
    
    print("AgentPrune example setup complete")


# Example 5: Using the AgentDropout strategy
async def dropout_example():
    """Example of using AgentDropout for agent skipping."""
    
    config = BEAMConfig(
        agents=[AgentConfig(name="Agent", count=5)],
        num_rounds=3,
        optimization=OptimizationConfig(
            strategy="dropout",
            optimize_spatial=True,
        )
    )
    
    dropout = AgentDropout(config)
    
    # After training, dropout learns which agents to skip
    # This reduces token usage while maintaining quality
    
    print("AgentDropout example setup complete")


# Example 6: Full workflow
async def full_workflow_example():
    """Complete example workflow."""
    
    # 1. Create configuration
    config = BEAMConfig(
        agents=[
            AgentConfig(name="Analyst", count=2),
            AgentConfig(name="Solver", count=2),
        ],
        num_rounds=2,
        optimization=OptimizationConfig(
            strategy="prune",
            optimize_spatial=True,
            pruning_rate=0.2,
        )
    )
    
    # 2. Create prompt templates
    prompts = PromptSet(name="problem_solving")
    prompts.add_role(
        "Analyst",
        system="You analyze problems and identify key information.",
        user="Analyze: {task}\n\nContext: {context}"
    )
    prompts.add_role(
        "Solver",
        system="You solve problems based on analysis.",
        user="Problem: {task}\n\nAnalysis: {context}"
    )
    
    # 3. Create agents (with mock execution for demo)
    def mock_execute(state):
        return f"Processed: {state['task'][:50]}..."
    
    agents = []
    for agent_config in config.agents:
        for i in range(agent_config.count):
            node = create_agent_node(
                role=agent_config.name,
                execute_fn=mock_execute
            )
            agents.append(node)
    
    print(f"Created {len(agents)} agents")
    print(f"Configuration: {config.optimization.strategy} strategy")
    print(f"Rounds: {config.num_rounds}")
    
    # 4. In real usage, you would:
    # - Set up the graph with these agents
    # - Train on your dataset
    # - Run optimized inference


if __name__ == "__main__":
    asyncio.run(full_workflow_example())
