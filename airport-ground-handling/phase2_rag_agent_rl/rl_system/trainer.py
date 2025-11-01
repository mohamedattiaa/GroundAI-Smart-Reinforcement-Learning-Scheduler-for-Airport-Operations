"""
RL Training Module
"""

import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
from pathlib import Path
from typing import Dict, Optional
import matplotlib.pyplot as plt


class TrainingCallback(BaseCallback):
    """Custom callback for training monitoring"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_delays = []
    
    def _on_step(self) -> bool:
        """Called at each step"""
        
        # Check if episode ended
        if self.locals.get('dones'):
            for i, done in enumerate(self.locals['dones']):
                if done:
                    info = self.locals['infos'][i]
                    self.episode_delays.append(info.get('total_delay', 0))
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of a rollout"""
        
        if len(self.episode_delays) > 0:
            avg_delay = np.mean(self.episode_delays[-100:])
            self.logger.record('custom/avg_delay', avg_delay)
            
            if self.verbose > 0:
                print(f"Average delay (last 100 episodes): {avg_delay:.2f}")


class RLTrainer:
    """Train RL agents for airport scheduling"""
    
    def __init__(
        self,
        env,
        algorithm: str = 'PPO',
        device: str = 'auto'
    ):
        """
        Initialize trainer
        
        Args:
            env: Gymnasium environment
            algorithm: RL algorithm ('PPO', 'DQN', 'A2C')
            device: Device for training ('auto', 'cpu', 'cuda')
        """
        self.env = env
        self.algorithm = algorithm
        self.device = device
        
        # Wrap environment
        self.vec_env = DummyVecEnv([lambda: Monitor(env)])
        
        # Initialize model
        self.model = self._create_model()
        
        print(f"✅ Initialized {algorithm} trainer")
        print(f"   Device: {self.device}")
    
    def _create_model(self):
        """Create RL model"""
        
        if self.algorithm == 'PPO':
            model = PPO(
                'MultiInputPolicy',
                self.vec_env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                verbose=1,
                device=self.device,
                tensorboard_log="logs/tensorboard"
            )
        
        elif self.algorithm == 'DQN':
            model = DQN(
                'MultiInputPolicy',
                self.vec_env,
                learning_rate=1e-4,
                buffer_size=50000,
                learning_starts=1000,
                batch_size=32,
                gamma=0.99,
                verbose=1,
                device=self.device,
                tensorboard_log="logs/tensorboard"
            )
        
        elif self.algorithm == 'A2C':
            model = A2C(
                'MultiInputPolicy',
                self.vec_env,
                learning_rate=7e-4,
                n_steps=5,
                gamma=0.99,
                verbose=1,
                device=self.device,
                tensorboard_log="logs/tensorboard"
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        return model
    
    def train(
        self,
        total_timesteps: int = 100000,
        callback=None,
        save_freq: int = 10000,
        save_path: str = "models/rl_checkpoints"
    ):
        """
        Train the model
        
        Args:
            total_timesteps: Total training timesteps
            callback: Custom callback
            save_freq: Save model every N steps
            save_path: Path to save checkpoints
        """
        print(f"\n{'='*60}")
        print(f"TRAINING {self.algorithm}")
        print(f"{'='*60}")
        print(f"Total timesteps: {total_timesteps:,}")
        print(f"Save frequency: {save_freq:,}")
        print()
        
        # Create save directory
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Setup callback
        if callback is None:
            callback = TrainingCallback(verbose=1)
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save final model
        final_path = f"{save_path}/{self.algorithm}_final.zip"
        self.model.save(final_path)
        print(f"\n✅ Training complete! Model saved to {final_path}")
    
    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False
    ) -> Dict:
        """
        Evaluate trained model
        
        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render environment
        
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING MODEL")
        print(f"{'='*60}")
        
        episode_rewards = []
        episode_delays = []
        episode_lengths = []
        
        for ep in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
                
                if render:
                    print(self.env.render())
            
            episode_rewards.append(episode_reward)
            episode_delays.append(info['total_delay'])
            episode_lengths.append(episode_length)
            
            print(f"Episode {ep+1}/{n_episodes}: "
                  f"Reward={episode_reward:.2f}, "
                  f"Delay={info['total_delay']:.2f}, "
                  f"Length={episode_length}")
        
        results = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_delay': np.mean(episode_delays),
            'std_delay': np.std(episode_delays),
            'mean_length': np.mean(episode_lengths)
        }
        
        print(f"\n{'='*60}")
        print("EVALUATION RESULTS")
        print(f"{'='*60}")
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean Delay: {results['mean_delay']:.2f} ± {results['std_delay']:.2f}")
        print(f"Mean Episode Length: {results['mean_length']:.1f}")
        
        return results
    
    def save(self, path: str):
        """Save model"""
        self.model.save(path)
        print(f"✅ Model saved to {path}")
    
    def load(self, path: str):
        """Load model"""
        if self.algorithm == 'PPO':
            self.model = PPO.load(path, env=self.vec_env)
        elif self.algorithm == 'DQN':
            self.model = DQN.load(path, env=self.vec_env)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(path, env=self.vec_env)
        
        print(f"✅ Model loaded from {path}")