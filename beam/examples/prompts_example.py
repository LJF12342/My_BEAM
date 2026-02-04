"""
Example: Using BEAM's prompt template system.

This example shows how to create and manage prompt templates
for different domains and use cases.
"""

from beam import PromptRegistry, PromptSet, PromptTemplate


# Example 1: Create a prompt set for math problem solving
def create_math_prompts():
    """Create prompts for mathematical problem solving."""
    
    prompts = PromptSet(
        name="math",
        description="Prompts for mathematical problem solving with multiple agents"
    )
    
    # Add solver role
    prompts.add_role(
        role="solver",
        system="""You are an expert mathematician. Your task is to solve mathematical problems step by step.

Guidelines:
1. Read the problem carefully
2. Identify the key information and what is being asked
3. Show your work step by step
4. Verify your answer
5. State the final answer clearly""",
        user="""Problem: {task}

{context}

Please solve this problem step by step and provide the final answer."""
    )
    
    # Add verifier role
    prompts.add_role(
        role="verifier",
        system="""You are a mathematical verifier. Your task is to check solutions for correctness.

Guidelines:
1. Review the solution carefully
2. Check each step for errors
3. Verify the final answer
4. If incorrect, explain the error and provide the correct solution""",
        user="""Problem: {task}

Solutions to verify:
{context}

Please verify these solutions and confirm or correct the answer."""
    )
    
    # Add decision template
    prompts.set_decision_template(
        system="""You are a decision agent that synthesizes multiple mathematical solutions.
Review all provided solutions and determine the correct final answer.""",
        user="""Problem: {task}

Agent solutions:
{context}

Based on the above solutions, provide the final answer (just the number/result):"""
    )
    
    return prompts


# Example 2: Create prompts for code generation
def create_coding_prompts():
    """Create prompts for code generation tasks."""
    
    prompts = PromptSet(
        name="coding",
        description="Prompts for collaborative code generation"
    )
    
    prompts.add_role(
        role="architect",
        system="""You are a software architect. Design the high-level structure and approach for solving coding problems.

Focus on:
- Algorithm selection
- Data structure choices
- Edge cases to consider
- Time/space complexity""",
        user="""Task: {task}

{context}

Provide your architectural analysis and recommended approach."""
    )
    
    prompts.add_role(
        role="coder",
        system="""You are an expert programmer. Write clean, efficient, well-documented code.

Guidelines:
- Follow best practices
- Handle edge cases
- Write readable code
- Include brief comments for complex logic""",
        user="""Task: {task}

{context}

Write the code solution."""
    )
    
    prompts.add_role(
        role="reviewer",
        system="""You are a code reviewer. Review code for correctness, efficiency, and best practices.

Check for:
- Bugs and logic errors
- Edge case handling
- Code style and readability
- Potential optimizations""",
        user="""Task: {task}

Code to review:
{context}

Provide your review and any suggested improvements."""
    )
    
    prompts.set_decision_template(
        system="Synthesize the code solutions and reviews into a final, correct implementation.",
        user="""Task: {task}

Solutions and reviews:
{context}

Provide the final code solution:"""
    )
    
    return prompts


# Example 3: Register prompts with the registry
def setup_prompts():
    """Register all prompt sets with the global registry."""
    
    # Register math prompts
    math_prompts = create_math_prompts()
    PromptRegistry.register("math", math_prompts)
    
    # Register coding prompts
    coding_prompts = create_coding_prompts()
    PromptRegistry.register("coding", coding_prompts)
    
    print(f"Registered prompt sets: {PromptRegistry.keys()}")


# Example 4: Using prompts with agents
def example_usage():
    """Demonstrate how to use prompts with BEAM agents."""
    
    from beam import create_agent_node, LLMRegistry
    
    # Setup prompts
    setup_prompts()
    
    # Get the math prompt set
    math_prompts = PromptRegistry.get("math")
    
    # Get prompts for a specific role
    system, user_template = math_prompts.get_prompt(
        "solver",
        task="What is 15% of 80?",
        context=""
    )
    
    print("System prompt:")
    print(system)
    print("\nUser prompt:")
    print(user_template)
    
    # Create an agent node with these prompts
    # (Assuming you have an LLM configured)
    # llm = LLMRegistry.get("gpt-4o")
    # 
    # solver_node = create_agent_node(
    #     role="solver",
    #     llm=llm,
    #     system_prompt=system,
    #     user_prompt_template="Problem: {task}\n\n{context}\n\nSolve step by step:"
    # )


# Example 5: Save and load prompts
def save_load_example():
    """Show how to save and load prompt sets."""
    
    prompts = create_math_prompts()
    
    # Save to file
    prompts.save("math_prompts.json")
    print("Saved prompts to math_prompts.json")
    
    # Load from file
    loaded = PromptSet.load("math_prompts.json")
    print(f"Loaded prompt set: {loaded.name}")
    print(f"Available roles: {loaded.get_roles()}")


if __name__ == "__main__":
    example_usage()
