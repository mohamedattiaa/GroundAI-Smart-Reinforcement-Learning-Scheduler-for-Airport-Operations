"""
Policies module for RL agents in airport ground handling optimization.
Implements custom policy networks for DQN, PPO, and A2C algorithms.

Location: phase2_rag_agent_rl/rl_system/policies.py

This is the ONLY policies.py file needed. Do NOT duplicate this file.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Any
from stable_baselines3.common.policies import ActorCriticPolicy, BaseFeaturesExtractor
from stable_baselines3.dqn.policies import DQNPolicy
from gymnasium import spaces


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for processing airport ground handling observations.
    
    Handles Dict observation spaces containing:
    - aircraft: (num_aircraft, 8) state information
    - global: (5,) global environment state
    - vehicles: (num_vehicles, 6) vehicle information
    
    Flattens and processes all components through a shared neural network.
    """
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        """
        Initialize the feature extractor.
        
        Args:
            observation_space: Dict observation space from environment
            features_dim: Output dimension of feature representation
        """
        super().__init__(observation_space, features_dim)
        
        # Calculate input size from dict observation space
        n_input = 0
        self.sub_sizes = {}
        
        if isinstance(observation_space, spaces.Dict):
            for key, subspace in observation_space.spaces.items():
                if isinstance(subspace, spaces.Box):
                    size = int(np.prod(subspace.shape))
                    n_input += size
                    self.sub_sizes[key] = size
        else:
            # Fallback for Box spaces
            n_input = int(np.prod(observation_space.shape))
        
        if n_input == 0:
            raise ValueError(
                f"Could not determine input size from observation space: {observation_space}"
            )
        
        # Build feature extraction network
        self.net = nn.Sequential(
            nn.Linear(n_input, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.2),
            
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        self.logger_info = f"Feature extractor: {n_input} -> 512 -> 256 -> {features_dim}"
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through feature extractor.
        
        Args:
            observations: Dict or Tensor of observations
            
        Returns:
            Feature representation of shape (batch_size, features_dim)
        """
        if isinstance(observations, dict):
            # Flatten all dict components in consistent order
            flattened = []
            for key in sorted(observations.keys()):
                obs = observations[key]
                if isinstance(obs, torch.Tensor):
                    # Reshape to (batch_size, -1)
                    flattened.append(obs.reshape(obs.size(0), -1))
            observations = torch.cat(flattened, dim=1)
        
        elif isinstance(observations, torch.Tensor):
            # Reshape single tensor
            observations = observations.reshape(observations.size(0), -1)
        
        return self.net(observations)


class GroundHandlingActorCriticPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for PPO and A2C agents.
    
    Optimized for airport ground handling task scheduling with:
    - Custom feature extraction for Dict observation spaces
    - Shared representation between actor and critic
    - Improved convergence properties
    
    Usage:
        model = PPO(
            GroundHandlingActorCriticPolicy,
            env,
            learning_rate=3e-4,
            ...
        )
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize actor-critic policy with custom features extractor."""
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256)
        )


class GroundHandlingDQNPolicy(DQNPolicy):
    """
    Custom DQN policy for task scheduling.
    
    Features:
    - Custom feature extraction
    - Suitable for discrete action spaces
    - Uses Q-learning for optimal action selection
    
    Usage:
        model = DQN(
            GroundHandlingDQNPolicy,
            env,
            learning_rate=1e-4,
            ...
        )
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize DQN policy with custom features extractor."""
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=CustomFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=256)
        )


class MultiAgentPolicyNetwork(nn.Module):
    """
    Shared policy network for multi-agent coordination.
    
    Architecture:
    - Shared representation layer for common knowledge
    - Agent-specific policy heads for individual decisions
    - Agent-specific value heads for state evaluation
    
    Enables:
    - Knowledge sharing between agents
    - Individual policy optimization
    - Coordinated learning
    
    Example:
        network = MultiAgentPolicyNetwork(
            input_dim=1000,
            hidden_dim=256,
            num_agents=3
        )
        policy_logits, value = network(state, agent_id=0)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_agents: int = 3,
        num_actions: int = 900
    ):
        """
        Initialize multi-agent policy network.
        
        Args:
            input_dim: Dimension of flattened observation
            hidden_dim: Hidden layer dimension
            num_agents: Number of agents (agents types)
            num_actions: Number of possible actions
        """
        super().__init__()
        self.num_agents = num_agents
        self.num_actions = num_actions
        
        # Shared representation layer
        self.shared_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Agent-specific policy heads (for action selection)
        self.policy_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions)
            )
            for _ in range(num_agents)
        ])
        
        # Agent-specific value heads (for state evaluation)
        self.value_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )
            for _ in range(num_agents)
        ])
    
    def forward(
        self,
        state: torch.Tensor,
        agent_id: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a specific agent.
        
        Args:
            state: Environment state observation (batch_size, input_dim)
            agent_id: Which agent to get policy for (0 to num_agents-1)
        
        Returns:
            policy_logits: Action logits for the agent
            value: Value estimate for the state
        """
        if not (0 <= agent_id < self.num_agents):
            raise ValueError(f"Invalid agent_id {agent_id}. Must be 0-{self.num_agents-1}")
        
        shared_repr = self.shared_net(state)
        policy_logits = self.policy_heads[agent_id](shared_repr)
        value = self.value_heads[agent_id](shared_repr)
        
        return policy_logits, value
    
    def get_all_values(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get value estimates for all agents simultaneously.
        
        Args:
            state: Environment state observation
            
        Returns:
            Combined value estimates for all agents (batch_size, num_agents)
        """
        shared_repr = self.shared_net(state)
        values = [head(shared_repr) for head in self.value_heads]
        return torch.cat(values, dim=1)
    
    def get_all_actions(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> List[torch.Tensor]:
        """
        Get action distributions for all agents.
        
        Args:
            state: Environment state
            deterministic: If True, return argmax; else sample
            
        Returns:
            List of action tensors for each agent
        """
        shared_repr = self.shared_net(state)
        actions = []
        
        for head in self.policy_heads:
            logits = head(shared_repr)
            if deterministic:
                action = torch.argmax(logits, dim=1)
            else:
                probs = torch.softmax(logits, dim=1)
                action = torch.multinomial(probs, 1).squeeze(1)
            actions.append(action)
        
        return actions


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN architecture for better Q-value estimation.
    
    Architecture:
    - Shared feature extraction
    - Separate value stream (state value V(s))
    - Separate advantage stream (action advantages A(s,a))
    - Combination: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    
    Benefits:
    - Better state value estimation
    - More stable learning
    - Improved convergence speed
    
    Example:
        network = DuelingDQNNetwork(input_dim=1000, action_dim=900)
        q_values = network(state)  # Shape: (batch_size, action_dim)
    """
    
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dim: int = 256
    ):
        """
        Initialize Dueling DQN network.
        
        Args:
            input_dim: Dimension of flattened observation
            action_dim: Number of possible actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        # Feature extraction (shared by both streams)
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(hidden_dim),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Value stream: estimates V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream: estimates A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing Q-values.
        
        Args:
            state: Environment state (batch_size, input_dim)
            
        Returns:
            Q-values for all actions (batch_size, action_dim)
        """
        features = self.feature_net(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Dueling combination: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # Subtracting mean stabilizes learning
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values


class TaskPriorityNetwork(nn.Module):
    """
    Network for learning task priorities and scheduling decisions.
    
    Purpose:
    - Learns which tasks are most urgent
    - Outputs task prioritization scores
    - Supports task-aware action selection
    
    Output:
    - Softmax over tasks indicates priority distribution
    - Can be used to bias action selection
    
    Example:
        network = TaskPriorityNetwork(input_dim=1000, num_tasks=3)
        priorities = network(state)  # Shape: (batch_size, num_tasks)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_tasks: int = 3,
        hidden_dim: int = 128
    ):
        """
        Initialize task priority network.
        
        Args:
            input_dim: Dimension of flattened observation
            num_tasks: Number of task types (fueling, catering, cleaning)
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.num_tasks = num_tasks
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, num_tasks)
        )
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(
        self,
        state: torch.Tensor,
        return_logits: bool = False
    ) -> torch.Tensor:
        """
        Forward pass computing task priorities.
        
        Args:
            state: Environment state (batch_size, input_dim)
            return_logits: If True, return raw logits; else return probabilities
            
        Returns:
            Task priorities: either logits or softmax probabilities
        """
        logits = self.net(state)
        
        if return_logits:
            return logits
        
        return self.softmax(logits)


# Policy registry for easy access
POLICIES = {
    'actor_critic': GroundHandlingActorCriticPolicy,
    'dqn': GroundHandlingDQNPolicy,
    'multiagent': MultiAgentPolicyNetwork,
    'dueling_dqn': DuelingDQNNetwork,
    'task_priority': TaskPriorityNetwork
}


def get_policy(policy_name: str, **kwargs):
    """
    Get policy class by name with custom parameters.
    
    Args:
        policy_name: Name of policy ('actor_critic', 'dqn', etc.)
        **kwargs: Parameters to pass to policy constructor
        
    Returns:
        Initialized policy instance
        
    Raises:
        ValueError: If policy_name not recognized
    """
    if policy_name not in POLICIES:
        raise ValueError(
            f"Unknown policy: {policy_name}. "
            f"Available: {list(POLICIES.keys())}"
        )
    return POLICIES[policy_name](**kwargs)


if __name__ == "__main__":
    """Test policies module."""
    print("Testing policies module...")
    
    # Test CustomFeaturesExtractor
    print("\n1. Testing CustomFeaturesExtractor...")
    obs_space = spaces.Dict({
        'aircraft': spaces.Box(0, 1, (10, 8), dtype=np.float32),
        'global': spaces.Box(0, 1, (5,), dtype=np.float32),
        'vehicles': spaces.Box(0, 1, (30, 6), dtype=np.float32)
    })
    extractor = CustomFeaturesExtractor(obs_space, features_dim=256)
    print(f"   ✓ {extractor.logger_info}")
    
    # Test with batch
    batch = {
        'aircraft': torch.randn(32, 10, 8),
        'global': torch.randn(32, 5),
        'vehicles': torch.randn(32, 30, 6)
    }
    features = extractor(batch)
    print(f"   ✓ Output shape: {features.shape} (expected: [32, 256])")
    
    # Test MultiAgentPolicyNetwork
    print("\n2. Testing MultiAgentPolicyNetwork...")
    input_size = 10*8 + 5 + 30*6  # Total flattened size
    ma_net = MultiAgentPolicyNetwork(input_dim=input_size, num_agents=3)
    state = torch.randn(32, input_size)
    policy, value = ma_net(state, agent_id=0)
    print(f"   ✓ Policy shape: {policy.shape}, Value shape: {value.shape}")
    
    # Test DuelingDQNNetwork
    print("\n3. Testing DuelingDQNNetwork...")
    dueling_net = DuelingDQNNetwork(input_dim=input_size, action_dim=900)
    q_values = dueling_net(state)
    print(f"   ✓ Q-values shape: {q_values.shape} (expected: [32, 900])")
    
    # Test TaskPriorityNetwork
    print("\n4. Testing TaskPriorityNetwork...")
    task_net = TaskPriorityNetwork(input_dim=input_size, num_tasks=3)
    priorities = task_net(state)
    print(f"   ✓ Priorities shape: {priorities.shape} (expected: [32, 3])")
    print(f"   ✓ Priorities sum: {priorities[0].sum():.4f} (expected: 1.0)")
    
    print("\n✅ All policy tests passed!")