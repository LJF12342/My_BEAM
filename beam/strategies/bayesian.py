"""AgentBayesian strategy - Bayesian optimization with optional MCMC sampling."""

from typing import Dict, Any, List, Optional, Tuple
import torch
import asyncio
import copy
import numpy as np

from beam.core.graph import AgentGraph
from beam.core.config import BEAMConfig

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import pyro
    from pyro.infer import MCMC, NUTS
    from pyro.distributions import Normal, Bernoulli
    HAS_PYRO = True
except ImportError:
    HAS_PYRO = False


class AgentBayesian:
    """
    AgentBayesian optimization strategy.
    
    This strategy uses Bayesian optimization to learn edge importance,
    with optional MCMC sampling for more robust uncertainty estimation.
    
    Features:
    - Gaussian Process regression for edge weight estimation
    - Optional MCMC sampling using NUTS for better exploration
    - Uncertainty-aware edge pruning
    
    Example:
        ```python
        config = BEAMConfig(
            agents=[AgentConfig(name="MathSolver", count=4)],
            optimization=OptimizationConfig(
                strategy=OptimizationStrategy.BAYESIAN,
                use_bayesian=True,
                use_mcmc=True,
                mcmc_samples=50
            )
        )
        
        bayesian = AgentBayesian(config)
        bayesian.set_graph(graph)
        
        # Train with Bayesian optimization
        bayesian.train(train_data, eval_fn)
        
        # Inference
        result = await bayesian.run({"task": "..."})
        ```
    """

    def __init__(self, config: BEAMConfig):
        self.config = config
        self.graph: Optional[AgentGraph] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.training_history: List[Dict[str, Any]] = []
        
        # Bayesian-specific parameters
        self.use_mcmc = config.optimization.use_mcmc
        self.mcmc_samples = config.optimization.mcmc_samples
        self.mcmc_warmup = config.optimization.mcmc_warmup
        
        # Gaussian Process for edge weight estimation
        self.gp: Optional[GaussianProcessRegressor] = None
        
        # Edge statistics
        self.edge_means: Optional[torch.Tensor] = None
        self.edge_vars: Optional[torch.Tensor] = None
        self.temporal_edge_means: Optional[torch.Tensor] = None
        self.temporal_edge_vars: Optional[torch.Tensor] = None
        
        # Observation history for GP
        self.observation_history: List[Dict[str, Any]] = []

    def set_graph(self, graph: AgentGraph):
        """Set the agent graph to optimize."""
        self.graph = graph
        n = len(graph.agent_names)
        num_edges = n * n
        
        # Initialize edge statistics
        self.edge_means = torch.zeros(num_edges)
        self.edge_vars = torch.ones(num_edges)
        self.temporal_edge_means = torch.zeros(num_edges)
        self.temporal_edge_vars = torch.ones(num_edges)
        
        # Initialize Gaussian Process
        if HAS_SKLEARN:
            kernel = RBF() + WhiteKernel()
            self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5)
        
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

    def _update_edge_statistics(self, observations: List[Dict[str, Any]]):
        """Update edge mean and variance estimates using observations."""
        if not observations or self.edge_means is None:
            return
        
        # Collect edge activations and rewards
        edge_rewards = {}
        
        for obs in observations:
            edges = obs.get("active_edges", [])
            reward = obs.get("reward", 0.0)
            
            for edge_idx in edges:
                if edge_idx not in edge_rewards:
                    edge_rewards[edge_idx] = []
                edge_rewards[edge_idx].append(reward)
        
        # Update means and variances
        for edge_idx, rewards in edge_rewards.items():
            if edge_idx < len(self.edge_means):
                # Bayesian update
                prior_mean = self.edge_means[edge_idx].item()
                prior_var = self.edge_vars[edge_idx].item()
                
                obs_mean = np.mean(rewards)
                obs_var = np.var(rewards) + 1e-6
                n_obs = len(rewards)
                
                # Posterior update
                posterior_var = 1.0 / (1.0 / prior_var + n_obs / obs_var)
                posterior_mean = posterior_var * (prior_mean / prior_var + n_obs * obs_mean / obs_var)
                
                self.edge_means[edge_idx] = posterior_mean
                self.edge_vars[edge_idx] = posterior_var

    def _mcmc_sample_edges(
        self,
        round_idx: int,
        temperature: float = 1.0
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Sample edge connections using MCMC.
        
        Returns:
            Tuple of (active_edge_indices, log_probability)
        """
        if not HAS_PYRO or self.graph is None:
            return [], torch.tensor(0.0)
        
        n = len(self.graph.agent_names)
        num_edges = n * n
        
        # Define Bayesian model
        def model():
            edge_means = pyro.sample(
                "edge_means",
                Normal(
                    torch.zeros(num_edges),
                    torch.ones(num_edges)
                )
            )
            
            active_edges = []
            for i in range(num_edges):
                prob = torch.sigmoid(edge_means[i] / temperature)
                is_active = pyro.sample(
                    f"edge_{i}",
                    Bernoulli(prob)
                )
                if is_active:
                    active_edges.append(i)
            
            return edge_means, active_edges
        
        # Run MCMC
        nuts_kernel = NUTS(model, step_size=0.01)
        mcmc = MCMC(nuts_kernel, num_samples=self.mcmc_samples, warmup_steps=self.mcmc_warmup)
        mcmc.run()
        
        # Get samples
        samples = mcmc.get_samples()
        edge_means_samples = samples["edge_means"]
        
        # Use mean of samples
        mean_edge_means = edge_means_samples.mean(dim=0)
        
        # Determine active edges
        active_edges = []
        log_prob = torch.tensor(0.0)
        
        for i in range(num_edges):
            prob = torch.sigmoid(mean_edge_means[i] / temperature)
            if torch.rand(1) < prob:
                active_edges.append(i)
                log_prob = log_prob + torch.log(prob)
            else:
                log_prob = log_prob + torch.log(1 - prob)
        
        return active_edges, log_prob

    def _gp_sample_edges(
        self,
        temperature: float = 1.0
    ) -> Tuple[List[int], torch.Tensor]:
        """
        Sample edge connections using Gaussian Process.
        
        Returns:
            Tuple of (active_edge_indices, log_probability)
        """
        if self.edge_means is None or self.edge_vars is None:
            return [], torch.tensor(0.0)
        
        active_edges = []
        log_prob = torch.tensor(0.0)
        
        for i in range(len(self.edge_means)):
            # Sample from posterior
            mean = self.edge_means[i]
            std = torch.sqrt(self.edge_vars[i])
            
            # UCB-style exploration
            sampled_value = mean + 0.5 * std * torch.randn(1).item()
            prob = torch.sigmoid(torch.tensor(sampled_value / temperature))
            
            if torch.rand(1) < prob:
                active_edges.append(i)
                log_prob = log_prob + torch.log(prob)
            else:
                log_prob = log_prob + torch.log(1 - prob)
        
        return active_edges, log_prob

    def train(
        self,
        train_data: List[Dict[str, Any]],
        eval_fn,
        num_iterations: Optional[int] = None,
        batch_size: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the Bayesian strategy.
        
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
            "edge_uncertainty": []
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
                
                # Track uncertainty
                if self.edge_vars is not None:
                    avg_uncertainty = self.edge_vars.mean().item()
                    stats["edge_uncertainty"].append(avg_uncertainty)
                
                accuracy = stats["total_solved"] / stats["total_executed"]
                stats["accuracies"].append(accuracy)
                
                if verbose:
                    uncertainty = stats["edge_uncertainty"][-1] if stats["edge_uncertainty"] else 0
                    print(f"  Batch {batch_idx // batch_size + 1}: "
                          f"Acc={accuracy:.4f}, Loss={batch_stats['loss']:.4f}, "
                          f"Uncertainty={uncertainty:.4f}")
        
        self.training_history.append(stats)
        return stats

    async def _train_batch(
        self,
        batch: List[Dict[str, Any]],
        eval_fn
    ) -> Dict[str, Any]:
        """Train on a single batch with Bayesian optimization."""
        tasks = []
        answers = []
        observations = []
        
        for record in batch:
            task = record["task"]
            answer = record["answer"]
            answers.append(answer)
            
            input_dict = {"task": task}
            
            # Sample edges using Bayesian method
            if self.use_mcmc and HAS_PYRO:
                active_edges, log_prob = self._mcmc_sample_edges(0)
            else:
                active_edges, log_prob = self._gp_sample_edges()
            
            tasks.append(self._run_with_edges(input_dict, active_edges))
            observations.append({
                "active_edges": active_edges,
                "log_prob": log_prob
            })
        
        results = await asyncio.gather(*tasks)
        
        loss_list = []
        solved = 0
        
        for i, ((raw_answer, _), true_answer) in enumerate(zip(results, answers)):
            prediction = raw_answer[0] if raw_answer else ""
            utility = eval_fn(prediction, true_answer)
            
            if utility > 0:
                solved += 1
            
            # Update observation with reward
            observations[i]["reward"] = utility
            
            # Policy gradient loss
            log_prob = observations[i]["log_prob"]
            single_loss = -log_prob * utility
            loss_list.append(single_loss)
        
        # Update edge statistics with observations
        self._update_edge_statistics(observations)
        self.observation_history.extend(observations)
        
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

    async def _run_with_edges(
        self,
        inputs: Dict[str, Any],
        active_edges: List[int]
    ) -> Tuple[List[Any], torch.Tensor]:
        """Run inference with specified active edges."""
        if self.graph is None:
            return [], torch.tensor(0.0)
        
        n = len(self.graph.agent_names)
        
        # Set masks based on active edges
        mask = torch.zeros(n * n)
        for edge_idx in active_edges:
            if edge_idx < len(mask):
                mask[edge_idx] = 1.0
        
        # Temporarily override masks
        original_masks = self.graph.spatial_masks.data.clone()
        self.graph.spatial_masks.data = mask
        
        try:
            answers, log_probs = await self.graph.arun(inputs, self.config.num_rounds)
        finally:
            # Restore original masks
            self.graph.spatial_masks.data = original_masks
        
        return answers, log_probs

    async def run(
        self,
        inputs: Dict[str, Any],
        num_rounds: Optional[int] = None,
        use_mean: bool = True
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Run optimized inference.
        
        Args:
            inputs: Input dictionary with 'task' key
            num_rounds: Number of inference rounds
            use_mean: Use mean edge weights (vs sampling)
            
        Returns:
            Tuple of (answers, metadata)
        """
        if self.graph is None:
            raise ValueError("Graph must be set before running")
        
        num_rounds = num_rounds or self.config.num_rounds
        
        if use_mean and self.edge_means is not None:
            # Use learned mean edge weights
            n = len(self.graph.agent_names)
            active_edges = []
            
            for i in range(len(self.edge_means)):
                if self.edge_means[i] > 0:  # Positive mean = likely useful
                    active_edges.append(i)
            
            answers, log_probs = await self._run_with_edges(inputs, active_edges)
        else:
            # Sample edges
            if self.use_mcmc and HAS_PYRO:
                active_edges, _ = self._mcmc_sample_edges(0)
            else:
                active_edges, _ = self._gp_sample_edges()
            
            answers, log_probs = await self._run_with_edges(inputs, active_edges)
        
        metadata = {
            "log_probs": log_probs.item() if isinstance(log_probs, torch.Tensor) else log_probs,
            "active_edges": len(active_edges) if 'active_edges' in dir() else 0,
            "use_mcmc": self.use_mcmc,
            "strategy": "bayesian"
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

    def get_edge_statistics(self) -> Dict[str, Any]:
        """Get statistics about edge weights."""
        if self.edge_means is None or self.edge_vars is None:
            return {}
        
        return {
            "spatial": {
                "mean": self.edge_means.mean().item(),
                "std": self.edge_means.std().item(),
                "uncertainty_mean": self.edge_vars.mean().item(),
                "positive_edges": (self.edge_means > 0).sum().item()
            },
            "temporal": {
                "mean": self.temporal_edge_means.mean().item() if self.temporal_edge_means is not None else 0,
                "std": self.temporal_edge_means.std().item() if self.temporal_edge_means is not None else 0,
                "uncertainty_mean": self.temporal_edge_vars.mean().item() if self.temporal_edge_vars is not None else 0
            },
            "observations": len(self.observation_history)
        }

    def save(self, path: str):
        """Save strategy state."""
        state = {
            "edge_means": self.edge_means,
            "edge_vars": self.edge_vars,
            "temporal_edge_means": self.temporal_edge_means,
            "temporal_edge_vars": self.temporal_edge_vars,
            "observation_history": self.observation_history,
            "training_history": self.training_history
        }
        torch.save(state, path)

    def load(self, path: str):
        """Load strategy state."""
        state = torch.load(path)
        self.edge_means = state.get("edge_means")
        self.edge_vars = state.get("edge_vars")
        self.temporal_edge_means = state.get("temporal_edge_means")
        self.temporal_edge_vars = state.get("temporal_edge_vars")
        self.observation_history = state.get("observation_history", [])
        self.training_history = state.get("training_history", [])
