"""
Advanced Trainer for Extended RL Training Sessions
Supports longer training, curriculum learning, and hyperparameter optimization.

Location: airport-ground-handling/advanced_trainer.py (root level for easy access)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from datetime import datetime, timedelta
import json

import numpy as np
import torch
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
    StopTrainingOnRewardThreshold,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from phase2_rag_agent_rl.rl_system.environment import AirportGroundHandlingEnv
from phase2_rag_agent_rl.rl_system.policies import (
    GroundHandlingActorCriticPolicy,
    GroundHandlingDQNPolicy
)


class TrainingMetricsCallback(BaseCallback):
    """Track and log detailed training metrics."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_tasks = []
        self.episode_delays = []
        self.start_time = None
        self.best_mean_reward = float('-inf')
    
    def _on_training_start(self) -> None:
        self.start_time = datetime.now()
    
    def _on_step(self) -> bool:
        # This is called after every env.step()
        # Use it to track progress
        if self.num_timesteps % 10000 == 0:
            elapsed = datetime.now() - self.start_time
            fps = self.num_timesteps / elapsed.total_seconds()
            self.logger.record("time/fps", fps)
        
        return True
    
    def _on_training_end(self) -> None:
        elapsed = datetime.now() - self.start_time
        self.logger.record("time/total_training_time", elapsed.total_seconds())


class AdvancedTrainer:
    """Advanced trainer with extended training capabilities."""
    
    def __init__(self, verbose: int = 1):
        """Initialize advanced trainer."""
        self.verbose = verbose
        self._setup_logging()
        self._create_directories()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"AdvancedTrainer initialized (device: {self.device})")
    
    def _setup_logging(self):
        """Setup logging."""
        log_dir = Path('./logs')
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('AdvancedTrainer')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        fh = logging.FileHandler(log_dir / 'advanced_training.log', encoding='utf-8')
        ch = logging.StreamHandler()
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _create_directories(self):
        """Create necessary directories."""
        for d in ['checkpoints', 'models', 'logs', 'videos', 'training_logs']:
            Path(d).mkdir(exist_ok=True)
    
    def create_environment(self, num_envs: int = 1):
        """Create environment(s)."""
        def make_env():
            return AirportGroundHandlingEnv(
                num_aircraft=10,
                num_vehicles=30,
                episode_length=100,
            )
        
        if num_envs == 1:
            return make_env()
        else:
            env = DummyVecEnv([make_env for _ in range(num_envs)])
            env = VecNormalize(env, norm_obs=True, norm_reward=True)
            return env
    
    def train_extended(
        self,
        total_timesteps: int = 500000,
        algorithm: str = 'PPO',
        num_envs: int = 1,
        save_interval: int = 25000,
        eval_interval: int = 25000,
        **algo_kwargs
    ) -> str:
        """
        Train for extended timesteps with detailed logging.
        
        Args:
            total_timesteps: Total training steps
            algorithm: 'PPO', 'A2C', or 'DQN'
            num_envs: Number of parallel environments
            save_interval: Save checkpoint every N steps
            eval_interval: Evaluate every N steps
            **algo_kwargs: Algorithm-specific hyperparameters
        
        Returns:
            Path to best model
        """
        # Validate algorithm
        if algorithm not in ['PPO', 'A2C', 'DQN']:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        if algorithm == 'DQN':
            self.logger.warning("⚠️  DQN requires Discrete action space")
            self.logger.warning("   Your environment has MultiDiscrete([10, 3, 30])")
            self.logger.warning("   Recommend: Use PPO instead")
            self.logger.info("   To use DQN, you would need to modify environment.py")
            raise ValueError(
                "DQN not supported for MultiDiscrete action spaces. "
                "Use PPO or A2C instead."
            )
        
        self.logger.info("="*70)
        self.logger.info(f"STARTING EXTENDED {algorithm} TRAINING")
        self.logger.info("="*70)
        self.logger.info(f"Total timesteps: {total_timesteps:,}")
        self.logger.info(f"Parallel environments: {num_envs}")
        self.logger.info(f"Device: {self.device}")
        
        # Create environment
        env = self.create_environment(num_envs)
        
        # Set default hyperparameters
        if algorithm == 'PPO':
            defaults = {
                'learning_rate': 3e-4,
                'n_steps': 2048,
                'batch_size': 64,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
            }
        elif algorithm == 'DQN':
            defaults = {
                'learning_rate': 1e-4,
                'buffer_size': 1000000,
                'learning_starts': 1000,
                'batch_size': 32,
                'gamma': 0.99,
            }
        elif algorithm == 'A2C':
            defaults = {
                'learning_rate': 7e-4,
                'n_steps': 5,
                'gamma': 0.99,
                'gae_lambda': 0.95,
            }
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        defaults.update(algo_kwargs)
        
        self.logger.info(f"Hyperparameters: {defaults}")
        
        # Create model
        if algorithm == 'PPO':
            model = PPO(
                GroundHandlingActorCriticPolicy,
                env,
                device=self.device,
                verbose=self.verbose,
                tensorboard_log='./logs',
                **defaults
            )
        elif algorithm == 'DQN':
            model = DQN(
                GroundHandlingDQNPolicy,
                env,
                device=self.device,
                verbose=self.verbose,
                tensorboard_log='./logs',
                **defaults
            )
        else:  # A2C
            model = A2C(
                GroundHandlingActorCriticPolicy,
                env,
                device=self.device,
                verbose=self.verbose,
                tensorboard_log='./logs',
                **defaults
            )
        
        # Setup callbacks
        callbacks = self._create_callbacks(algorithm, save_interval, eval_interval)
        
        # Train
        self.logger.info("Starting training loop...")
        start_time = datetime.now()
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                callback=callbacks,
                progress_bar=True,
                log_interval=10
            )
            self.logger.info("[OK] Training completed successfully")
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training error: {e}", exc_info=True)
            raise
        finally:
            elapsed = datetime.now() - start_time
            self.logger.info(f"Training time: {elapsed}")
            env.close()
        
        # Save final model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models/{algorithm.lower()}_extended_{timestamp}"
        model.save(model_path)
        self.logger.info(f"[OK] Final model saved: {model_path}")
        
        return model_path
    
    def _create_callbacks(self, algorithm: str, save_interval: int, 
                         eval_interval: int):
        """Create training callbacks."""
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_interval,
            save_path='./checkpoints',
            name_prefix=algorithm.lower()
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        eval_env = self.create_environment()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./models',
            log_path='./logs',
            eval_freq=eval_interval,
            n_eval_episodes=10,
            deterministic=True,
            render=False
        )
        callbacks.append(eval_callback)
        
        # Metrics callback
        metrics_callback = TrainingMetricsCallback()
        callbacks.append(metrics_callback)
        
        return callbacks
    
    def continuous_learning(
        self,
        model_path: str,
        additional_timesteps: int = 100000,
        **kwargs
    ) -> str:
        """
        Continue training from an existing model.
        
        Args:
            model_path: Path to existing model
            additional_timesteps: Additional training steps
            **kwargs: Algorithm hyperparameters
        
        Returns:
            Path to newly trained model
        """
        self.logger.info("="*70)
        self.logger.info("RESUMING TRAINING FROM CHECKPOINT")
        self.logger.info("="*70)
        self.logger.info(f"Loading model: {model_path}")
        
        env = self.create_environment()
        
        # Determine algorithm type
        if 'ppo' in model_path.lower():
            model = PPO.load(model_path, env=env, device=self.device)
            algo = 'PPO'
        elif 'dqn' in model_path.lower():
            model = DQN.load(model_path, env=env, device=self.device)
            algo = 'DQN'
        else:
            model = A2C.load(model_path, env=env, device=self.device)
            algo = 'A2C'
        
        self.logger.info(f"Model loaded: {algo}")
        self.logger.info(f"Additional timesteps: {additional_timesteps:,}")
        
        # Setup callbacks
        callbacks = self._create_callbacks(algo, 25000, 25000)
        
        # Continue training
        start_time = datetime.now()
        try:
            model.learn(
                total_timesteps=additional_timesteps,
                callback=callbacks,
                progress_bar=True,
                reset_num_timesteps=False  # Continue counting
            )
            self.logger.info("[OK] Continued training completed")
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
        finally:
            elapsed = datetime.now() - start_time
            self.logger.info(f"Training time: {elapsed}")
            env.close()
        
        # Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models/{algo.lower()}_continued_{timestamp}"
        model.save(model_path)
        self.logger.info(f"[OK] Model saved: {model_path}")
        
        return model_path
    
    def evaluate_detailed(
        self,
        model_path: str,
        num_episodes: int = 20,
        save_stats: bool = True
    ) -> Dict[str, Any]:
        """
        Detailed evaluation with comprehensive statistics.
        
        Args:
            model_path: Path to model
            num_episodes: Number of evaluation episodes
            save_stats: Save stats to JSON file
        
        Returns:
            Detailed statistics dictionary
        """
        self.logger.info("="*70)
        self.logger.info("DETAILED EVALUATION")
        self.logger.info("="*70)
        
        env = self.create_environment()
        
        # Load model
        if 'dqn' in model_path.lower():
            model = DQN.load(model_path, env=env, device=self.device)
        elif 'a2c' in model_path.lower():
            model = A2C.load(model_path, env=env, device=self.device)
        else:
            model = PPO.load(model_path, env=env, device=self.device)
        
        # Evaluate
        stats = {
            'episode_rewards': [],
            'episode_lengths': [],
            'episode_delays': [],
            'episode_tasks': [],
            'episode_success': []
        }
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            
            ep_reward = 0
            ep_length = 0
            ep_delay = 0
            ep_tasks = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                ep_reward += reward
                ep_delay += info.get('delay', 0)
                ep_tasks += info.get('tasks_completed', 0)
                ep_length += 1
            
            success = 1 if ep_reward > 0 else 0
            stats['episode_rewards'].append(ep_reward)
            stats['episode_lengths'].append(ep_length)
            stats['episode_delays'].append(ep_delay)
            stats['episode_tasks'].append(ep_tasks)
            stats['episode_success'].append(success)
            
            self.logger.info(
                f"Episode {ep+1:2d}: R={ep_reward:7.2f}, T={ep_length:3d}, "
                f"D={ep_delay:8.2f}, Tasks={ep_tasks:3.0f}"
            )
        
        env.close()
        
        # Calculate summary statistics
        summary = {
            'num_episodes': num_episodes,
            'mean_reward': float(np.mean(stats['episode_rewards'])),
            'std_reward': float(np.std(stats['episode_rewards'])),
            'max_reward': float(np.max(stats['episode_rewards'])),
            'min_reward': float(np.min(stats['episode_rewards'])),
            'mean_length': float(np.mean(stats['episode_lengths'])),
            'mean_delay': float(np.mean(stats['episode_delays'])),
            'mean_tasks': float(np.mean(stats['episode_tasks'])),
            'success_rate': float(np.mean(stats['episode_success'])),
            'timestamp': datetime.now().isoformat(),
            'model': model_path
        }
        
        self.logger.info("\n" + "="*70)
        self.logger.info("EVALUATION SUMMARY")
        self.logger.info("="*70)
        for key, value in summary.items():
            if key not in ['timestamp', 'model']:
                self.logger.info(f"{key:20s}: {value:10.4f}")
        
        # Save stats
        if save_stats:
            stats_file = f"./training_logs/eval_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(stats_file, 'w') as f:
                json.dump({**summary, 'detailed': stats}, f, indent=2)
            self.logger.info(f"[OK] Stats saved: {stats_file}")
        
        return summary