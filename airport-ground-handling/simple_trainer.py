"""
Simplified Trainer for quick testing and training without YAML dependency.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime

import numpy as np
import torch
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from phase2_rag_agent_rl.rl_system.environment import AirportGroundHandlingEnv
from phase2_rag_agent_rl.rl_system.policies import (
    GroundHandlingActorCriticPolicy,
    GroundHandlingDQNPolicy
)


class SimpleTrainer:
    """Simplified trainer without YAML dependencies."""
    
    def __init__(self, verbose: int = 1):
        """Initialize trainer with defaults."""
        self.verbose = verbose
        self._setup_logging()
        self._create_directories()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"SimpleTrainer initialized (device: {self.device})")
    
    def _setup_logging(self):
        """Setup logging."""
        log_dir = Path('./logs')
        log_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger('SimpleTrainer')
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        
        fh = logging.FileHandler(log_dir / 'training.log', encoding='utf-8')
        ch = logging.StreamHandler()
        ch.stream.reconfigure(encoding='utf-8') if hasattr(ch.stream, 'reconfigure') else None
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def _create_directories(self):
        """Create necessary directories."""
        for d in ['checkpoints', 'models', 'logs', 'videos']:
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
    
    def train_ppo(
        self,
        total_timesteps: int = 50000,
        num_envs: int = 1,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_epochs: int = 10,
    ) -> str:
        """Train PPO agent."""
        self.logger.info("="*70)
        self.logger.info("Starting PPO Training")
        self.logger.info("="*70)
        self.logger.info(f"Total timesteps: {total_timesteps}")
        self.logger.info(f"Learning rate: {learning_rate}")
        self.logger.info(f"Batch size: {batch_size}")
        
        env = self.create_environment(num_envs)
        
        model = PPO(
            GroundHandlingActorCriticPolicy,
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            device=self.device,
            verbose=self.verbose,
            tensorboard_log='./logs'
        )
        
        # Setup callbacks
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path='./checkpoints',
            name_prefix='ppo'
        )
        
        eval_env = self.create_environment()
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path='./models',
            log_path='./logs',
            eval_freq=5000,
            n_eval_episodes=5,
            deterministic=True,
            render=False
        )
        
        try:
            self.logger.info("Starting learning loop...")
            model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback, eval_callback],
                progress_bar=True,
                log_interval=10
            )
            self.logger.info("[OK] Training completed successfully")
        except KeyboardInterrupt:
            self.logger.info("Training interrupted")
        except Exception as e:
            self.logger.error(f"Training error: {e}", exc_info=True)
            raise
        finally:
            env.close()
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models/ppo_model_{timestamp}"
        model.save(model_path)
        self.logger.info(f"[OK] Model saved: {model_path}")
        
        return model_path
    
    def train_dqn(
        self,
        total_timesteps: int = 50000,
        learning_rate: float = 1e-4,
        buffer_size: int = 100000,
        batch_size: int = 32,
    ) -> str:
        """Train DQN agent."""
        self.logger.info("="*70)
        self.logger.info("Starting DQN Training")
        self.logger.info("="*70)
        
        env = self.create_environment(1)
        
        model = DQN(
            GroundHandlingDQNPolicy,
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=1000,
            batch_size=batch_size,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_final_eps=0.05,
            device=self.device,
            verbose=self.verbose,
            tensorboard_log='./logs'
        )
        
        try:
            model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True,
                log_interval=10
            )
            self.logger.info("✓ Training completed")
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
        finally:
            env.close()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"./models/dqn_model_{timestamp}"
        model.save(model_path)
        self.logger.info(f"✓ Model saved: {model_path}")
        
        return model_path
    
    def evaluate(self, model_path: str, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate trained model."""
        self.logger.info("="*70)
        self.logger.info("Starting Evaluation")
        self.logger.info("="*70)
        
        env = self.create_environment()
        
        # Load model
        if 'dqn' in model_path.lower():
            model = DQN.load(model_path, env=env, device=self.device)
        elif 'a2c' in model_path.lower():
            model = A2C.load(model_path, env=env, device=self.device)
        else:
            model = PPO.load(model_path, env=env, device=self.device)
        
        metrics = {
            'rewards': [],
            'delays': [],
            'tasks': []
        }
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            ep_reward = 0
            ep_delay = 0
            ep_tasks = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_delay += info.get('delay', 0)
                ep_tasks += info.get('tasks_completed', 0)
            
            metrics['rewards'].append(ep_reward)
            metrics['delays'].append(ep_delay)
            metrics['tasks'].append(ep_tasks)
            
            self.logger.info(f"Episode {ep+1}: R={ep_reward:.2f}, D={ep_delay:.2f}, T={ep_tasks}")
        
        env.close()
        
        stats = {
            'mean_reward': np.mean(metrics['rewards']),
            'std_reward': np.std(metrics['rewards']),
            'mean_delay': np.mean(metrics['delays']),
            'mean_tasks': np.mean(metrics['tasks']),
        }
        
        self.logger.info("Evaluation Results:")
        for k, v in stats.items():
            self.logger.info(f"  {k}: {v:.2f}")
        
        return stats