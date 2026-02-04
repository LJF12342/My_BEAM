"""Base agent node implementation for BEAM toolkit."""

import shortuuid
from typing import List, Any, Optional, Dict, Callable, Union
from abc import ABC
import asyncio


class AgentNode(ABC):
    """
    Base class for agent nodes in the BEAM graph.
    
    Each node represents an agent that can process inputs and produce outputs.
    Nodes can have spatial (same round) and temporal (cross-round) connections.
    
    This is a flexible base class that can be used with:
    - Custom execution functions
    - LLM-based generation with prompt templates
    - LangChain/LangGraph integrations
    
    Attributes:
        id: Unique identifier for the node
        agent_name: Name/type of the agent
        role: Role description for the agent
        domain: Task domain
        llm: LLM instance for generation (optional)
        execute_fn: Custom execution function (optional)
    """

    def __init__(
        self,
        id: Optional[str] = None,
        agent_name: str = "",
        role: str = "",
        domain: str = "",
        llm: Optional[Any] = None,
        execute_fn: Optional[Callable] = None,
        system_prompt: str = "",
        user_prompt_template: str = "{task}\n{context}",
    ):
        """
        Initialize an agent node.
        
        Args:
            id: Optional node ID
            agent_name: Name for this agent
            role: Role description
            domain: Task domain
            llm: LLM instance with gen/agen methods
            execute_fn: Custom execution function (state) -> output
            system_prompt: System prompt for LLM-based execution
            user_prompt_template: User prompt template with {task}, {context} placeholders
        """
        self.id: str = id if id is not None else shortuuid.ShortUUID().random(length=4)
        self.agent_name: str = agent_name
        self.role: str = role
        self.domain: str = domain
        self.llm = llm
        self.execute_fn = execute_fn
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template
        
        # Connection lists
        self.spatial_predecessors: List["AgentNode"] = []
        self.spatial_successors: List["AgentNode"] = []
        self.temporal_predecessors: List["AgentNode"] = []
        self.temporal_successors: List["AgentNode"] = []
        
        # I/O
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        
        # Memory
        self.last_memory: Dict[str, List[Any]] = {
            'inputs': [],
            'outputs': [],
            'raw_inputs': []
        }

    @property
    def node_name(self) -> str:
        return self.__class__.__name__

    def add_predecessor(self, node: "AgentNode", connection_type: str = 'spatial'):
        """Add a predecessor node."""
        if connection_type == 'spatial' and node not in self.spatial_predecessors:
            self.spatial_predecessors.append(node)
            node.spatial_successors.append(self)
        elif connection_type == 'temporal' and node not in self.temporal_predecessors:
            self.temporal_predecessors.append(node)
            node.temporal_successors.append(self)

    def add_successor(self, node: "AgentNode", connection_type: str = 'spatial'):
        """Add a successor node."""
        if connection_type == 'spatial' and node not in self.spatial_successors:
            self.spatial_successors.append(node)
            node.spatial_predecessors.append(self)
        elif connection_type == 'temporal' and node not in self.temporal_successors:
            self.temporal_successors.append(node)
            node.temporal_predecessors.append(self)

    def remove_predecessor(self, node: "AgentNode", connection_type: str = 'spatial'):
        """Remove a predecessor node."""
        if connection_type == 'spatial' and node in self.spatial_predecessors:
            self.spatial_predecessors.remove(node)
            node.spatial_successors.remove(self)
        elif connection_type == 'temporal' and node in self.temporal_predecessors:
            self.temporal_predecessors.remove(node)
            node.temporal_successors.remove(self)

    def remove_successor(self, node: "AgentNode", connection_type: str = 'spatial'):
        """Remove a successor node."""
        if connection_type == 'spatial' and node in self.spatial_successors:
            self.spatial_successors.remove(node)
            node.spatial_predecessors.remove(self)
        elif connection_type == 'temporal' and node in self.temporal_successors:
            self.temporal_successors.remove(node)
            node.temporal_predecessors.remove(self)

    def clear_connections(self):
        """Clear all connections."""
        self.spatial_predecessors = []
        self.spatial_successors = []
        self.temporal_predecessors = []
        self.temporal_successors = []

    def update_memory(self):
        """Save current state to memory for next round."""
        self.last_memory['inputs'] = self.inputs.copy()
        self.last_memory['outputs'] = self.outputs.copy()
        self.last_memory['raw_inputs'] = self.raw_inputs.copy()

    def get_spatial_info(self) -> Dict[str, Dict]:
        """Get information from spatial predecessors."""
        spatial_info = {}
        for predecessor in self.spatial_predecessors:
            outputs = predecessor.outputs
            if isinstance(outputs, list) and len(outputs) > 0:
                output = outputs[-1]
            elif isinstance(outputs, list) and len(outputs) == 0:
                continue
            else:
                output = outputs
            spatial_info[predecessor.id] = {
                "role": predecessor.role,
                "output": output
            }
        return spatial_info

    def get_temporal_info(self) -> Dict[str, Dict]:
        """Get information from temporal predecessors (previous round)."""
        temporal_info = {}
        for predecessor in self.temporal_predecessors:
            outputs = predecessor.last_memory['outputs']
            if isinstance(outputs, list) and len(outputs) > 0:
                output = outputs[-1]
            elif isinstance(outputs, list) and len(outputs) == 0:
                continue
            else:
                output = outputs
            temporal_info[predecessor.id] = {
                "role": predecessor.role,
                "output": output
            }
        return temporal_info

    def _build_context(
        self,
        spatial_info: Dict[str, Dict],
        temporal_info: Dict[str, Dict]
    ) -> str:
        """Build context string from predecessor outputs."""
        parts = []
        
        for node_id, info in spatial_info.items():
            output = info.get("output", "")
            if output and output != "None.":
                parts.append(f"[{info['role']}]: {output}")
        
        for node_id, info in temporal_info.items():
            output = info.get("output", "")
            if output and output != "None.":
                parts.append(f"[Previous {info['role']}]: {output}")
        
        return "\n\n".join(parts)

    def execute(self, input_data: Any, **kwargs) -> List[Any]:
        """Execute the node synchronously."""
        self.outputs = []
        spatial_info = self.get_spatial_info()
        temporal_info = self.get_temporal_info()
        
        result = self._execute(input_data, spatial_info, temporal_info, **kwargs)
        
        if not isinstance(result, list):
            result = [result]
        self.outputs.extend(result)
        return self.outputs

    async def async_execute(self, input_data: Any, **kwargs) -> List[Any]:
        """Execute the node asynchronously."""
        self.outputs = []
        spatial_info = self.get_spatial_info()
        temporal_info = self.get_temporal_info()
        
        result = await self._async_execute(input_data, spatial_info, temporal_info, **kwargs)
        
        if not isinstance(result, list):
            result = [result]
        self.outputs.extend(result)
        return self.outputs

    def _execute(
        self,
        input_data: Any,
        spatial_info: Dict[str, Dict],
        temporal_info: Dict[str, Dict],
        **kwargs
    ) -> Any:
        """
        Internal execution method.
        
        Uses execute_fn if provided, otherwise uses LLM with prompts.
        """
        # If custom execute function provided
        if self.execute_fn is not None:
            context = self._build_context(spatial_info, temporal_info)
            state = {
                "task": input_data.get("task", str(input_data)) if isinstance(input_data, dict) else str(input_data),
                "context": context,
                "spatial_info": spatial_info,
                "temporal_info": temporal_info,
                **kwargs
            }
            return self.execute_fn(state)
        
        # LLM-based execution
        if self.llm is not None:
            task = input_data.get("task", str(input_data)) if isinstance(input_data, dict) else str(input_data)
            context = self._build_context(spatial_info, temporal_info)
            
            user_prompt = self.user_prompt_template.format(
                task=task,
                context=context if context else "No additional context."
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return self.llm.gen(messages)
        
        # Fallback
        return f"Node {self.agent_name}: No execution method configured"

    async def _async_execute(
        self,
        input_data: Any,
        spatial_info: Dict[str, Dict],
        temporal_info: Dict[str, Dict],
        **kwargs
    ) -> Any:
        """
        Internal async execution method.
        
        Uses execute_fn if provided, otherwise uses LLM with prompts.
        """
        # If custom execute function provided
        if self.execute_fn is not None:
            context = self._build_context(spatial_info, temporal_info)
            state = {
                "task": input_data.get("task", str(input_data)) if isinstance(input_data, dict) else str(input_data),
                "context": context,
                "spatial_info": spatial_info,
                "temporal_info": temporal_info,
                **kwargs
            }
            
            if asyncio.iscoroutinefunction(self.execute_fn):
                return await self.execute_fn(state)
            else:
                return await asyncio.get_event_loop().run_in_executor(
                    None, lambda: self.execute_fn(state)
                )
        
        # LLM-based execution
        if self.llm is not None:
            task = input_data.get("task", str(input_data)) if isinstance(input_data, dict) else str(input_data)
            context = self._build_context(spatial_info, temporal_info)
            
            user_prompt = self.user_prompt_template.format(
                task=task,
                context=context if context else "No additional context."
            )
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            return await self.llm.agen(messages)
        
        # Fallback
        return f"Node {self.agent_name}: No execution method configured"


def create_agent_node(
    role: str,
    llm: Any = None,
    execute_fn: Callable = None,
    system_prompt: str = "",
    user_prompt_template: str = "Task: {task}\n\nContext:\n{context}",
    **kwargs
) -> AgentNode:
    """
    Factory function to create an agent node.
    
    Args:
        role: Role name for the agent
        llm: LLM instance (optional)
        execute_fn: Custom execution function (optional)
        system_prompt: System prompt for LLM
        user_prompt_template: User prompt template
        **kwargs: Additional node arguments
        
    Returns:
        Configured AgentNode
        
    Example:
        ```python
        # With LLM
        node = create_agent_node(
            role="Solver",
            llm=my_llm,
            system_prompt="You are a problem solver.",
            user_prompt_template="Solve: {task}\\nHints: {context}"
        )
        
        # With custom function
        def my_solver(state):
            return f"Solved: {state['task']}"
        
        node = create_agent_node(
            role="Solver",
            execute_fn=my_solver
        )
        ```
    """
    return AgentNode(
        agent_name=role,
        role=role,
        llm=llm,
        execute_fn=execute_fn,
        system_prompt=system_prompt,
        user_prompt_template=user_prompt_template,
        **kwargs
    )
