"""
Enhanced Reinforcement Learning Trainer for Airport Ground Handling Operations.
Supports PPO, DQN, and A2C algorithms with multi-agent coordination.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import warnings

import numpy as np
import torch

try:
    from stable_baselines3 import PPO, DQN, A2C
    from stable_baselines3.common.callbacks import (
        EvalCallback, StopTrainingOnRewardThreshold, 
        CheckpointCallback, BaseCallback
    )
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
except ImportError as e:
    warnings.warn(f"Could not import stable_baselines3: {e}")

try:
    from phase2_rag_agent_rl.rl_system.environment import AirportGroundHandlingEnv
    from phase2_rag_agent_rl.rl_system.policies import (
        GroundHandlingActorCriticPolicy,
        GroundHandlingDQNPolicy,
        MultiAgentPolicyNetwork
    )
except ImportError as e:
    warnings.warn(f"Could not import project modules: {e}")


class MetricsCallback(BaseCallback):
    """Custom callback to track detailed metrics during training."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_delays = []
        self.episode_lengths = []
        self.task_completions = []
        self.current_episode_reward = 0
        self.current_episode_delay = 0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        # Get current environment
        env = self.model.env
        
        # Check if episode is done
        dones = self.model.env.env_method("get_episode_done")
        for i, done in enumerate(dones):
            if done:
                # Extract metrics from environment
                info = self.model.env.env_method("get_episode_info")[i]
                self.episode_rewards.append(info.get('total_reward', 0))
                self.episode_delays.append(info.get('total_delay', 0))
                self.episode_lengths.append(info.get('episode_length', 0))
                self.task_completions.append(info.get('tasks_completed', 0))
                
                # Log to tensorboard
                if len(self.episode_rewards) % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    avg_delay = np.mean(self.episode_delays[-10:])
                    self.logger.record("metrics/avg_reward_10", avg_reward)
                    self.logger.record("metrics/avg_delay_10", avg_delay)
        
        return True


class RLTrainer:
    """Main trainer class for RL agents."""
    
    def __init__(self, config_path: str = "config.yaml", verbose: int = 1):
        """
        Initialize the RL Trainer.
        
        Args:
            config_path: Path to configuration file
            verbose: Verbosity level (0=no output, 1=progress, 2=detailed)
        """
        self.verbose = verbose
        self._setup_logging()  # Setup logging FIRST
        self.config = self._load_config(config_path)
        self.logger.info("RLTrainer initialized")
        
        # Create directories
        self._create_directories()
        
        # Device setup
        self.device = self._get_device()
        self.logger.info(f"Using device: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        if not Path(config_path).exists():
            if hasattr(self, 'logger'):
                self.logger.warning(f"Config file not found: {config_path}. Using defaults.")
            else:
                print(f"⚠️  Config file not found: {config_path}. Using defaults.")
            return self._get_default_config()
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            if hasattr(self, 'logger'):
                self.logger.info(f"Config loaded from {config_path}")
            return config
    
    def _get_default_config(self) -> Dict:
        """Return default configuration if file not found."""
        return {
            'environment': {'name': 'AirportGroundHandling-v1'},
            'training': {'algorithm': 'PPO', 'total_timesteps': 100000},
            'system': {'random_seed': 42}
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_dir = Path('./logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'airport_rl.log'
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler (supports unicode)
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # Console handler (safe for Windows)
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if self.verbose > 1 else logging.INFO)
        ch.stream.reconfigure(encoding='utf-8') if hasattr(ch.stream, 'reconfigure') else None
        
        # Formatter (no emojis for console)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(file_formatter)
        ch.setFormatter(console_formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _create_directories(self):
        """Create necessary directories."""
        dirs = [
            'checkpoints',
            'models',
            'logs',
            'videos',
            'experiments'
        ]
        for d in dirs:
            Path(d).mkdir(exist_ok=True)
    
    def _get_device(self) -> str:
        """Determine device to use."""
        device_config = self.config.get('system', {}).get('device', 'auto')
        if device_config == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_config
    
    def create_environment(self, num_envs: int = 1) -> Any:
        """
        Create training environment(s).
        
        Args:
            num_envs: Number of parallel environments
        
        Returns:
            Environment or VecEnv wrapper
        """
        self.logger.info(f"Creating {num_envs} environment(s)...")
        
        env_config = self.config.get('environment', {})
        
        def make_env():
            return AirportGroundHandlingEnv(
                num_aircraft=env_config.get('num_aircraft', 10),
                num_vehicles=env_config.get('num_vehicles', 30),
                episode_length=env_config.get('episode_length', 100),
                reward_config=env_config.get('reward', {}),
                task_config=env_config.get('tasks', {})
            )
        
        if num_envs == 1:
            env = make_env()
        else:
            env = make_vec_env(make_env, n_envs=num_envs, vec_env_cls=DummyVecEnv)
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        self.logger.info("[OK] Environment(s) created successfully")
        return env
    
    def train(self, num_envs: int = 1, resume_from: Optional[str] = None) -> str:
        """
        Train the RL agent.
        
        Args:
            num_envs: Number of parallel environments
            resume_from: Path to checkpoint to resume from
        
        Returns:
            Path to saved model
        """
        self.logger.info("=" * 70)
        self.logger.info("STARTING TRAINING")
        self.logger.info("=" * 70)
        
        # Create environment
        env = self.create_environment(num_envs)
        
        # Get training config
        train_config = self.config.get('training', {})
        algorithm = train_config.get('algorithm', 'PPO').upper()
        total_timesteps = train_config.get('total_timesteps', 100000)
        
        self.logger.info(f"Algorithm: {algorithm}")
        self.logger.info(f"Total timesteps: {total_timesteps}")
        self.logger.info(f"Number of parallel envs: {num_envs}")
        
        # Create model
        if resume_from and Path(resume_from).exists():
            self.logger.info(f"Resuming from checkpoint: {resume_from}")
            model = self._load_model(algorithm, env, resume_from)
        else:
            model = self._create_model(algorithm, env, train_config)
        
        # Setup callbacks
        callbacks = self._setup_callbacks(train_config)
        
        # Train
        try:
            self.logger.info("Starting training loop...")
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=self.verbose > 0,
                log_interval=10
            )
            self.logger.info("[OK] Training completed successfully")
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}", exc_info=True)
            raise
        
        # Save model
        model_path = self._save_model(model, algorithm)
        env.close()
        
        return model_path
    
    def _create_model(self, algorithm: str, env: Any, config: Dict) -> Any:
        """Create RL model based on algorithm."""
        self.logger.info(f"Creating {algorithm} model...")
        
        algo_config = config.get(algorithm.lower(), {})
        
        base_params = {
            'env': env,
            'device': self.device,
            'tensorboard_log': config.get('tensorboard_log'),
            'verbose': self.verbose
        }
        
        if algorithm == 'PPO':
            base_params.update({
                'learning_rate': algo_config.get('learning_rate', 3e-4),
                'n_steps': algo_config.get('n_steps', 2048),
                'batch_size': algo_config.get('batch_size', 64),
                'n_epochs': algo_config.get('n_epochs', 10),
                'gamma': algo_config.get('gamma', 0.99),
                'gae_lambda': algo_config.get('gae_lambda', 0.95),
                'clip_range': algo_config.get('clip_range', 0.2),
                'ent_coef': algo_config.get('ent_coef', 0.01),
                'vf_coef': algo_config.get('vf_coef', 0.5),
            })
            model = PPO(GroundHandlingActorCriticPolicy, **base_params)
        
        elif algorithm == 'DQN':
            base_params.update({
                'learning_rate': algo_config.get('learning_rate', 1e-4),
                'buffer_size': algo_config.get('buffer_size', 100000),
                'learning_starts': algo_config.get('learning_starts', 1000),
                'batch_size': algo_config.get('batch_size', 32),
                'gamma': algo_config.get('gamma', 0.99),
                'exploration_fraction': algo_config.get('exploration_fraction', 0.1),
                'exploration_final_eps': algo_config.get('exploration_final_eps', 0.05),
            })
            model = DQN(GroundHandlingDQNPolicy, **base_params)
        
        elif algorithm == 'A2C':
            base_params.update({
                'learning_rate': algo_config.get('learning_rate', 7e-4),
                'n_steps': algo_config.get('n_steps', 5),
                'gamma': algo_config.get('gamma', 0.99),
                'gae_lambda': algo_config.get('gae_lambda', 0.95),
                'ent_coef': algo_config.get('ent_coef', 0.01),
                'vf_coef': algo_config.get('vf_coef', 0.5),
            })
            model = A2C(GroundHandlingActorCriticPolicy, **base_params)
        
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        self.logger.info(f"✓ {algorithm} model created")
        return model
    
    def _setup_callbacks(self, config: Dict) -> List[BaseCallback]:
        """Setup training callbacks."""
        callbacks = []
        
        try:
            # Checkpoint callback
            checkpoint_callback = CheckpointCallback(
                save_freq=config.get('save_freq', 10000),
                save_path='./checkpoints',
                name_prefix='model'
            )
            callbacks.append(checkpoint_callback)
            
            # Evaluation callback
            eval_env = self.create_environment()
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path='./models',
                log_path='./logs',
                eval_freq=config.get('eval_freq', 5000),
                n_eval_episodes=config.get('n_eval_episodes', 5),
                deterministic=True,
                render=False
            )
            callbacks.append(eval_callback)
            
            # Metrics callback
            metrics_callback = MetricsCallback()
            callbacks.append(metrics_callback)
        
        except Exception as e:
            self.logger.warning(f"Could not setup some callbacks: {e}")
        
        return callbacks
    
    def _save_model(self, model: Any, algorithm: str) -> str:
        """Save trained model."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models/{algorithm.lower()}_model_{timestamp}"
        model.save(model_path)
        self.logger.info(f"[OK] Model saved to: {model_path}")
        return model_path
    
    def _load_model(self, algorithm: str, env: Any, path: str) -> Any:
        """Load a trained model."""
        algo_class = {'ppo': PPO, 'dqn': DQN, 'a2c': A2C}[algorithm.lower()]
        return algo_class.load(path, env=env, device=self.device)
    
    def evaluate(self, model_path: str, num_episodes: int = 10, 
                render: bool = False) -> Dict[str, float]:
        """
        Evaluate a trained model.
        
        Args:
            model_path: Path to saved model
            num_episodes: Number of evaluation episodes
            render: Whether to render episodes
        
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("=" * 70)
        self.logger.info("STARTING EVALUATION")
        self.logger.info("=" * 70)
        
        env = self.create_environment()
        
        # Determine algorithm from model name
        if 'dqn' in model_path.lower():
            model = DQN.load(model_path, env=env, device=self.device)
        elif 'a2c' in model_path.lower():
            model = A2C.load(model_path, env=env, device=self.device)
        else:
            model = PPO.load(model_path, env=env, device=self.device)
        
        # Evaluate
        metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_delays': [],
            'task_completions': []
        }
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            total_reward = 0
            total_delay = 0
            tasks_completed = 0
            steps = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                total_delay += info.get('delay', 0)
                tasks_completed += info.get('tasks_completed', 0)
                steps += 1
                
                if render:
                    env.render()
            
            metrics['episode_rewards'].append(total_reward)
            metrics['episode_lengths'].append(steps)
            metrics['episode_delays'].append(total_delay)
            metrics['task_completions'].append(tasks_completed)
            
            self.logger.info(
                f"Episode {ep+1}: Reward={total_reward:.2f}, "
                f"Delay={total_delay:.2f}, Tasks={tasks_completed}"
            )
        
        # Calculate statistics
        stats = {
            'mean_reward': np.mean(metrics['episode_rewards']),
            'std_reward': np.std(metrics['episode_rewards']),
            'mean_delay': np.mean(metrics['episode_delays']),
            'mean_tasks_completed': np.mean(metrics['task_completions']),
        }
        
        self.logger.info("Evaluation Statistics:")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value:.2f}")
        
        env.close()
        return stats