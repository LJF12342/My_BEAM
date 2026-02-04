"""AgentDropout strategy - Dynamic agent dropout during inference."""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn.functional as F
import asyncio
import copy
import random

from beam.core.graph import AgentGraph
from beam.core.config import BEAMConfig


class AgentDropout:
    """
    AgentDropout optimization strategy.
    
    This strategy learns which agents can be skipped during inference
    based on their contribution to the final answer. It uses a learned
    dropout policy to dynamically skip low-contribution agents.
    
    The approach learns per-round dropout logits that determine which
    agent to skip in each round, reducing token usage while maintaining
    answer quality.
    
    Example:
        ```python
        config = BEAMConfig(
            agents=[AgentConfig(name="MathSolver", count=5)],
            optimization=OptimizationConfig(
                strategy=OptimizationStrategy.DROPOUT,
                optimize_spatial=True,
                dropout_rate=0.2
            )
        )
        
        dropout = AgentDropout(config)
        dropout.set_graph(graph)
        
        # Train
        dropout.train(train_data, eval_fn)
        
        # Inference with learned dropout
        result = await dropout.run({"task": "..."})
        ```
    """

    def __init__(self, config: BEAMConfig):
        self.config = config
        self.graph: Optional[AgentGraph] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.training_history: List[Dict[str, Any]] = []
        
        # Dropout-specific parameters
        self.num_rounds = config.num_rounds
        self.num_agents = config.get_num_agents()
        
        # Learnable dropout logits per round
        self.spatial_logits_dropout: Optional[torch.nn.ParameterList] = None
        self.temporal_logits_dropout: Optional[torch.nn.ParameterList] = None
        
        # Skip decisions per round
        self.skip_nodes: List[int] = []

    def set_graph(self, graph: AgentGraph):
        """Set the agent graph to optimize."""
        self.graph = graph
        self.num_agents = len(graph.agent_names)
        
        # Initialize dropout logits
        init_logit = 0.0  # Start with 50% probability
        
        self.spatial_logits_dropout = torch.nn.ParameterList([
            torch.nn.Parameter(
                torch.ones(self.num_agents * self.num_agents) * init_logit,
                requires_grad=True
            )
            for _ in range(self.num_rounds)
        ])
        
        self.temporal_logits_dropout = torch.nn.ParameterList([
            torch.nn.Parameter(
                torch.ones(self.num_agents * self.num_agents) * init_logit,
                requires_grad=True
            )
            for _ in range(self.num_rounds - 1)
        ])
        
        self._setup_optimizer()

    def _setup_optimizer(self):
        """Setup PyTorch optimizer."""
        if self.spatial_logits_dropout is None:
            return
        
        params = list(self.spatial_logits_dropout.parameters())
        if self.temporal_logits_dropout:
            params += list(self.temporal_logits_dropout.parameters())
        
        if params:
            self.optimizer = torch.optim.Adam(
                params,
                lr=self.config.optimization.learning_rate
            )

    def _compute_skip_decision(self, round_idx: int) -> Tuple[int, torch.Tensor]:
        """
        Compute which agent to skip in a given round.
        
        Returns:
            Tuple of (skip_index, log_probability)
        """
        if self.spatial_logits_dropout is None or self.graph is None:
            return -1, torch.tensor(0.0)
        
        # Compute importance scores for each agent
        logits = self.spatial_logits_dropout[round_idx]
        n = self.num_agents
        
        # Sum incoming and outgoing edge logits for each agent
        agent_scores = []
        for i in range(n):
            # Outgoing edges (row i)
            out_score = logits[i * n:(i + 1) * n].sum()
            # Incoming edges (column i)
            in_score = sum(logits[j * n + i] for j in range(n))
            agent_scores.append(out_score + in_score)
        
        # Convert to probabilities (lower score = more likely to skip)
        scores_tensor = torch.stack(agent_scores)
        skip_probs = F.softmax(-scores_tensor, dim=0)  # Negative because lower = skip
        
        # Sample which agent to skip
        skip_idx = torch.multinomial(skip_probs, num_samples=1).item()
        
        # Compute log probability for policy gradient
        log_prob = torch.log(skip_probs[skip_idx])
        
        return skip_idx, log_prob

    def train(
        self,
        train_data: List[Dict[str, Any]],
        eval_fn,
        num_iterations: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the dropout strategy.
        
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
        
        stats = {
            "total_solved": 0,
            "total_executed": 0,
            "losses": [],
            "accuracies": [],
            "skip_counts": []
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
                stats["skip_counts"].append(batch_stats["skipped"])
                
                accuracy = stats["total_solved"] / stats["total_executed"]
                stats["accuracies"].append(accuracy)
                
                if verbose:
                    print(f"  Batch {batch_idx // batch_size + 1}: "
                          f"Acc={accuracy:.4f}, Loss={batch_stats['loss']:.4f}, "
                          f"Skipped={batch_stats['skipped']}")
        
        # After training, determine fixed skip decisions
        self._compute_fixed_skip_decisions()
        
        self.training_history.append(stats)
        return stats

    async def _train_batch(
        self,
        batch: List[Dict[str, Any]],
        eval_fn
    ) -> Dict[str, Any]:
        """Train on a single batch with dropout."""
        tasks = []
        answers = []
        skip_log_probs = []
        
        for record in batch:
            task = record["task"]
            answer = record["answer"]
            answers.append(answer)
            
            # Compute skip decisions for this sample
            sample_skip_log_prob = torch.tensor(0.0)
            sample_skip_nodes = []
            
            for round_idx in range(self.num_rounds):
                skip_idx, log_prob = self._compute_skip_decision(round_idx)
                sample_skip_nodes.append(skip_idx)
                sample_skip_log_prob = sample_skip_log_prob + log_prob
            
            skip_log_probs.append(sample_skip_log_prob)
            
            # Run with skip decisions
            input_dict = {"task": task}
            tasks.append(self._run_with_skip(input_dict, sample_skip_nodes))
        
        results = await asyncio.gather(*tasks)
        
        loss_list = []
        solved = 0
        total_skipped = 0
        
        for (raw_answer, _), true_answer, skip_log_prob in zip(results, answers, skip_log_probs):
            prediction = raw_answer[0] if raw_answer else ""
            utility = eval_fn(prediction, true_answer)
            
            if utility > 0:
                solved += 1
            
            # Policy gradient loss for dropout decisions
            single_loss = -skip_log_prob * utility
            loss_list.append(single_loss)
            total_skipped += self.num_rounds  # One skip per round
        
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
            "loss": loss_value,
            "skipped": total_skipped // len(batch) if batch else 0
        }

    async def _run_with_skip(
        self,
        inputs: Dict[str, Any],
        skip_nodes: List[int]
    ) -> Tuple[List[Any], torch.Tensor]:
        """Run inference with specified skip decisions."""
        if self.graph is None:
            return [], torch.tensor(0.0)
        
        log_probs = torch.tensor(0.0)
        
        for round_idx in range(self.num_rounds):
            log_probs = log_probs + self.graph.construct_spatial_connections()
            log_probs = log_probs + self.graph.construct_temporal_connections(round_idx)
            
            # Topological execution with skipping
            in_degree = {
                node_id: len(node.spatial_predecessors)
                for node_id, node in self.graph.nodes.items()
            }
            queue = [nid for nid, deg in in_degree.items() if deg == 0]
            
            while queue:
                current_id = queue.pop(0)
                current_idx = list(self.graph.nodes.keys()).index(current_id)
                
                # Check if this agent should be skipped
                if current_idx == skip_nodes[round_idx]:
                    self.graph.nodes[current_id].outputs = ['None.']
                else:
                    try:
                        await asyncio.wait_for(
                            self.graph.nodes[current_id].async_execute(inputs),
                            timeout=600
                        )
                    except Exception as e:
                        self.graph.nodes[current_id].outputs = [f"Error: {e}"]
                
                for successor in self.graph.nodes[current_id].spatial_successors:
                    if successor.id in self.graph.nodes:
                        in_degree[successor.id] -= 1
                        if in_degree[successor.id] == 0:
                            queue.append(successor.id)
            
            self.graph.update_memory()
        
        # Execute decision node
        self.graph.connect_decision_node()
        if self.graph.decision_node:
            await self.graph.decision_node.async_execute(inputs)
            final_answers = self.graph.decision_node.outputs
        else:
            final_answers = list(self.graph.nodes.values())[-1].outputs if self.graph.nodes else []
        
        if not final_answers:
            final_answers = ["No answer"]
        
        return final_answers, log_probs

    def _compute_fixed_skip_decisions(self):
        """Compute fixed skip decisions based on learned logits."""
        self.skip_nodes = []
        
        if self.spatial_logits_dropout is None:
            return
        
        for round_idx in range(self.num_rounds):
            logits = self.spatial_logits_dropout[round_idx]
            n = self.num_agents
            
            # Find agent with lowest total edge weight
            agent_scores = []
            for i in range(n):
                out_score = logits[i * n:(i + 1) * n].sum().item()
                in_score = sum(logits[j * n + i].item() for j in range(n))
                agent_scores.append(out_score + in_score)
            
            # Skip the agent with lowest score
            skip_idx = agent_scores.index(min(agent_scores))
            self.skip_nodes.append(skip_idx)

    async def run(
        self,
        inputs: Dict[str, Any],
        num_rounds: Optional[int] = None,
        use_learned_skip: bool = True
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Run optimized inference with dropout.
        
        Args:
            inputs: Input dictionary with 'task' key
            num_rounds: Number of inference rounds
            use_learned_skip: Whether to use learned skip decisions
            
        Returns:
            Tuple of (answers, metadata)
        """
        if self.graph is None:
            raise ValueError("Graph must be set before running")
        
        num_rounds = num_rounds or self.num_rounds
        
        if use_learned_skip and self.skip_nodes:
            skip_nodes = self.skip_nodes[:num_rounds]
        else:
            skip_nodes = [-1] * num_rounds  # No skipping
        
        answers, log_probs = await self._run_with_skip(inputs, skip_nodes)
        
        metadata = {
            "log_probs": log_probs.item() if isinstance(log_probs, torch.Tensor) else log_probs,
            "skip_nodes": skip_nodes,
            "num_agents": self.num_agents,
            "strategy": "dropout"
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

    def get_skip_statistics(self) -> Dict[str, Any]:
        """Get statistics about skip decisions."""
        return {
            "skip_nodes_per_round": self.skip_nodes,
            "total_skips": len(self.skip_nodes),
            "agents_per_round": self.num_agents
        }

    def save(self, path: str):
        """Save strategy state."""
        state = {
            "spatial_logits_dropout": [p.data for p in self.spatial_logits_dropout] if self.spatial_logits_dropout else None,
            "temporal_logits_dropout": [p.data for p in self.temporal_logits_dropout] if self.temporal_logits_dropout else None,
            "skip_nodes": self.skip_nodes,
            "training_history": self.training_history
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load strategy state."""
        state = torch.load(path)
        
        if state["spatial_logits_dropout"] and self.spatial_logits_dropout:
            for i, data in enumerate(state["spatial_logits_dropout"]):
                if i < len(self.spatial_logits_dropout):
                    self.spatial_logits_dropout[i].data = data
        
        if state["temporal_logits_dropout"] and self.temporal_logits_dropout:
            for i, data in enumerate(state["temporal_logits_dropout"]):
                if i < len(self.temporal_logits_dropout):
                    self.temporal_logits_dropout[i].data = data
        
        self.skip_nodes = state.get("skip_nodes", [])
        self.training_history = state.get("training_history", [])
