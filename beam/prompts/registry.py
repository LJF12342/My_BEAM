"""
Prompt registry for BEAM toolkit.

Provides a centralized registry for managing prompt sets across
different domains and use cases.
"""

from typing import Dict, Optional, List
from beam.prompts.template import PromptSet, PromptTemplate


class PromptRegistry:
    """
    Registry for prompt sets.
    
    Allows registration and retrieval of prompt sets by domain name.
    Supports both programmatic and file-based prompt definitions.
    
    Example:
        ```python
        # Register a prompt set
        math_prompts = PromptSet(name="math")
        math_prompts.add_role("solver", system="...", user="...")
        PromptRegistry.register("math", math_prompts)
        
        # Or use decorator
        @PromptRegistry.register_set("coding")
        def create_coding_prompts():
            prompts = PromptSet(name="coding")
            prompts.add_role("coder", system="...", user="...")
            return prompts
        
        # Retrieve
        prompts = PromptRegistry.get("math")
        system, user = prompts.get_prompt("solver", task="2+2")
        ```
    """
    
    _registry: Dict[str, PromptSet] = {}

    @classmethod
    def register(cls, name: str, prompt_set: PromptSet):
        """
        Register a prompt set.
        
        Args:
            name: Domain/use case name
            prompt_set: The PromptSet to register
        """
        cls._registry[name] = prompt_set

    @classmethod
    def register_set(cls, name: str):
        """
        Decorator to register a prompt set factory.
        
        Args:
            name: Domain/use case name
            
        Example:
            ```python
            @PromptRegistry.register_set("my_domain")
            def create_prompts():
                prompts = PromptSet(name="my_domain")
                # ... configure prompts
                return prompts
            ```
        """
        def decorator(factory_fn):
            prompt_set = factory_fn()
            cls._registry[name] = prompt_set
            return factory_fn
        return decorator

    @classmethod
    def get(cls, name: str) -> PromptSet:
        """
        Get a registered prompt set.
        
        Args:
            name: Domain/use case name
            
        Returns:
            The registered PromptSet
        """
        if name not in cls._registry:
            raise ValueError(
                f"Prompt set '{name}' not found. "
                f"Available: {list(cls._registry.keys())}"
            )
        return cls._registry[name]

    @classmethod
    def get_or_create(cls, name: str) -> PromptSet:
        """
        Get a prompt set or create an empty one.
        
        Args:
            name: Domain/use case name
            
        Returns:
            The PromptSet (existing or new)
        """
        if name not in cls._registry:
            cls._registry[name] = PromptSet(name=name)
        return cls._registry[name]

    @classmethod
    def keys(cls) -> List[str]:
        """Get all registered domain names."""
        return list(cls._registry.keys())

    @classmethod
    def items(cls):
        """Get all registered (name, prompt_set) pairs."""
        return list(cls._registry.items())

    @classmethod
    def load_from_file(cls, name: str, path: str):
        """
        Load and register a prompt set from JSON file.
        
        Args:
            name: Domain name to register under
            path: Path to JSON file
        """
        prompt_set = PromptSet.load(path)
        cls._registry[name] = prompt_set

    @classmethod
    def clear(cls):
        """Clear all registered prompt sets."""
        cls._registry.clear()


def create_default_prompt_set(
    domain: str = "general",
    roles: Optional[List[str]] = None
) -> PromptSet:
    """
    Create a default prompt set with basic templates.
    
    This provides sensible defaults that can be customized.
    
    Args:
        domain: Domain name
        roles: List of role names (defaults to ["agent"])
        
    Returns:
        A PromptSet with default templates
        
    Example:
        ```python
        # Create with defaults
        prompts = create_default_prompt_set("math", roles=["solver", "verifier"])
        
        # Customize specific roles
        prompts.add_role(
            "solver",
            system="You are a math expert...",
            user="Solve: {task}"
        )
        ```
    """
    roles = roles or ["agent"]
    
    prompt_set = PromptSet(
        name=domain,
        description=f"Default prompts for {domain}"
    )
    
    # Add default role templates
    for role in roles:
        prompt_set.add_role(
            role=role,
            system=(
                f"You are a helpful {role} working on {domain} tasks.\n"
                "Provide clear, accurate responses based on the given information."
            ),
            user=(
                "Task: {task}\n\n"
                "{context}"
            )
        )
    
    # Add default decision template
    prompt_set.set_decision_template(
        system=(
            "You are a decision-making agent. Review all provided responses "
            "and synthesize them into a single, accurate final answer."
        ),
        user=(
            "Original task: {task}\n\n"
            "Agent responses:\n{context}\n\n"
            "Provide the final answer:"
        )
    )
    
    return prompt_set
