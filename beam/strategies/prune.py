"""AgentPrune strategy - Edge pruning based on learned importance."""

from typing import Dict, Any, List, Optional, Tuple
import torch
import asyncio
import copy

from beam.core.graph import AgentGraph
from beam.core.config import BEAMConfig


class AgentPrune:
    """
    AgentPrune optimization strategy.
    
    This strategy learns the importance of edges between agents and prunes
    low-importance connections to reduce token usage during inference.
    
    The approach uses differentiable edge weights that are optimized via
    policy gradient, then pruned based on learned importance.
    
    Example:
        ```python
        config = BEAMConfig(
            agents=[AgentConfig(name="MathSolver", count=4)],
            optimization=OptimizationConfig(
                strategy=OptimizationStrategy.PRUNE,
                optimize_spatial=True,
                optimize_temporal=True,
                pruning_rate=0.25
            )
        )
        
        prune = AgentPrune(config)
        prune.set_graph(graph)
        
        # Train
        prune.train(train_data, eval_fn)
        
        # Inference
        result = await prune.run({"task": "..."})
        ```
    """

    def __init__(self, config: BEAMConfig):
        self.config = config
        self.graph: Optional[AgentGraph] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.training_history: List[Dict[str, Any]] = []

    def set_graph(self, graph: AgentGraph):
        """Set the agent graph to optimize."""
        self.graph = graph
        self._setup_optimizer()

    def _setup_optimizer(self):
        """Setup PyTorch optimizer."""
        if self.graph is None:
            return
        
        params = self.graph.get_trainable_parameters()
        if params:
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.config.optimization.learning_rate
            )

    def train(
        self,
        train_data: List[Dict[str, Any]],
        eval_fn,
        num_iterations: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the pruning strategy.
        
        Args:
            train_data: Training examples with 'task' and 'answer'
            eval_fn: Evaluation function (prediction, answer) -> utility
            num_iterations: Training iterations
            batch_size: Batch size
            verbose: Print progress
            
        Returns:
            Training statistics
        """
        if self.graph is None:
            raise ValueError("Graph must be set before training")
        
        num_iterations = num_iterations or self.config.optimization.num_iterations
        batch_size = batch_size or self.config.optimization.batch_size
        pruning_rate = self.config.optimization.pruning_rate
        
        stats = {
            "total_solved": 0,
            "total_executed": 0,
            "losses": [],
            "accuracies": [],
            "edge_counts": []
        }
        
        for iteration in range(num_iterations):
            if verbose:
                print(f"Iteration {iteration + 1}/{num_iterations}")
            
            for batch_idx in range(0, len(train_data), batch_size):
                batch = train_data[batch_idx:batch_idx + batch_size]
                batch_stats = asyncio.get_event_loop().run_until_complete(
                    self._train_batch(batch, eval_fn)
                )
                
                stats["total_solved"] += batch_stats["solved"]
                stats["total_executed"] += batch_stats["executed"]
                stats["losses"].append(batch_stats["loss"])
                stats["edge_counts"].append(self.graph.num_edges)
                
                accuracy = stats["total_solved"] / stats["total_executed"]
                stats["accuracies"].append(accuracy)
                
                if verbose:
                    print(f"  Batch {batch_idx // batch_size + 1}: "
                          f"Acc={accuracy:.4f}, Loss={batch_stats['loss']:.4f}, "
                          f"Edges={self.graph.num_edges}")
            
            # Prune after each iteration
            self.graph.update_masks(pruning_rate)
            
            if verbose:
                print(f"  After pruning: {self.graph.num_edges} edges remaining")
        
        self.training_history.append(stats)
        return stats

    async def _train_batch(
        self,
        batch: List[Dict[str, Any]],
        eval_fn
    ) -> Dict[str, Any]:
        """Train on a single batch."""
        tasks = []
        answers = []
        
        for record in batch:
            task = record["task"]
            answer = record["answer"]
            answers.append(answer)
            
            realized_graph = copy.deepcopy(self.graph)
            realized_graph.spatial_logits = self.graph.spatial_logits
            realized_graph.temporal_logits = self.graph.temporal_logits
            
            input_dict = {"task": task}
            tasks.append(realized_graph.arun(input_dict, self.config.num_rounds))
        
        results = await asyncio.gather(*tasks)
        
        loss_list = []
        solved = 0
        
        for (raw_answer, log_prob), true_answer in zip(results, answers):
            prediction = raw_answer[0] if raw_answer else ""
            utility = eval_fn(prediction, true_answer)
            
            if utility > 0:
                solved += 1
            
            single_loss = -log_prob * utility
            loss_list.append(single_loss)
        
        if loss_list and self.optimizer:
            total_loss = torch.mean(torch.stack(loss_list))
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            loss_value = total_loss.item()
        else:
            loss_value = 0.0
        
        return {
            "solved": solved,
            "executed": len(batch),
            "loss": loss_value
        }

    async def run(
        self,
        inputs: Dict[str, Any],
        num_rounds: Optional[int] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Run optimized inference."""
        if self.graph is None:
            raise ValueError("Graph must be set before running")
        
        num_rounds = num_rounds or self.config.num_rounds
        answers, log_probs = await self.graph.arun(inputs, num_rounds)
        
        metadata = {
            "log_probs": log_probs.item() if isinstance(log_probs, torch.Tensor) else log_probs,
            "num_edges": self.graph.num_edges,
            "num_nodes": self.graph.num_nodes,
            "strategy": "prune"
        }
        
        return answers, metadata

    def run_sync(
        self,
        inputs: Dict[str, Any],
        num_rounds: Optional[int] = None
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Synchronous inference."""
        return asyncio.get_event_loop().run_until_complete(
            self.run(inputs, num_rounds)
        )

    def get_pruned_edges(self) -> Dict[str, Any]:
        """Get information about pruned edges."""
        if self.graph is None:
            return {}
        
        n = len(self.graph.agent_names)
        spatial_masks = self.graph.spatial_masks.view(n, n)
        temporal_masks = self.graph.temporal_masks.view(n, n)
        
        return {
            "spatial_active": (spatial_masks > 0).sum().item(),
            "spatial_total": n * n,
            "temporal_active": (temporal_masks > 0).sum().item(),
            "temporal_total": n * n,
            "spatial_matrix": spatial_masks.tolist(),
            "temporal_matrix": temporal_masks.tolist()
        }

    def save(self, path: str):
        """Save strategy state."""
        state = {
            "spatial_logits": self.graph.spatial_logits.data if self.graph else None,
            "temporal_logits": self.graph.temporal_logits.data if self.graph else None,
            "spatial_masks": self.graph.spatial_masks.data if self.graph else None,
            "temporal_masks": self.graph.temporal_masks.data if self.graph else None,
            "training_history": self.training_history
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load strategy state."""
        state = torch.load(path)
        if self.graph:
            if state["spatial_logits"] is not None:
                self.graph.spatial_logits.data = state["spatial_logits"]
            if state["temporal_logits"] is not None:
                self.graph.temporal_logits.data = state["temporal_logits"]
            if state["spatial_masks"] is not None:
                self.graph.spatial_masks.data = state["spatial_masks"]
            if state["temporal_masks"] is not None:
                self.graph.temporal_masks.data = state["temporal_masks"]
        self.training_history = state.get("training_history", [])
