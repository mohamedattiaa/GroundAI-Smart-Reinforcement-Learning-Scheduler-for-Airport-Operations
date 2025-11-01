"""
Evaluation and analysis tools for RL agents
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import pandas as pd
from pathlib import Path


class RLEvaluator:
    """Evaluate and visualize RL agent performance"""
    
    def __init__(self, env, model):
        """
        Initialize evaluator
        
        Args:
            env: Gymnasium environment
            model: Trained RL model
        """
        self.env = env
        self.model = model
        self.results = []
    
    def run_episodes(
        self,
        n_episodes: int = 100,
        deterministic: bool = True
    ) -> List[Dict]:
        """
        Run multiple evaluation episodes
        
        Args:
            n_episodes: Number of episodes
            deterministic: Use deterministic policy
        
        Returns:
            List of episode results
        """
        print(f"\n🔄 Running {n_episodes} evaluation episodes...")
        
        self.results = []
        
        for ep in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            
            episode_data = {
                'episode': ep,
                'reward': 0,
                'delay': 0,
                'length': 0,
                'assignments': 0,
                'completed_aircraft': 0
            }
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                
                episode_data['reward'] += reward
                episode_data['length'] += 1
                done = terminated or truncated
            
            # Extract final info
            episode_data['delay'] = info.get('total_delay', 0)
            episode_data['assignments'] = info.get('assignments', 0)
            episode_data['completed_aircraft'] = info.get('completed_aircraft', 0)
            
            self.results.append(episode_data)
            
            if (ep + 1) % 10 == 0:
                print(f"  Episode {ep + 1}/{n_episodes} complete")
        
        print("✅ Evaluation complete")
        return self.results
    
    def compute_statistics(self) -> Dict:
        """Compute statistics from results"""
        
        if not self.results:
            return {}
        
        df = pd.DataFrame(self.results)
        
        stats = {
            'episodes': len(self.results),
            'mean_reward': df['reward'].mean(),
            'std_reward': df['reward'].std(),
            'min_reward': df['reward'].min(),
            'max_reward': df['reward'].max(),
            'mean_delay': df['delay'].mean(),
            'std_delay': df['delay'].std(),
            'min_delay': df['delay'].min(),
            'max_delay': df['delay'].max(),
            'mean_length': df['length'].mean(),
            'mean_assignments': df['assignments'].mean(),
            'mean_completed': df['completed_aircraft'].mean()
        }
        
        return stats
    
    def print_statistics(self):
        """Print statistics"""
        
        stats = self.compute_statistics()
        
        if not stats:
            print("❌ No results to display")
            return
        
        print("\n" + "="*60)
        print("EVALUATION STATISTICS")
        print("="*60)
        print(f"Episodes: {stats['episodes']}")
        print(f"\nReward:")
        print(f"  Mean: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"  Min/Max: {stats['min_reward']:.2f} / {stats['max_reward']:.2f}")
        print(f"\nDelay:")
        print(f"  Mean: {stats['mean_delay']:.2f} ± {stats['std_delay']:.2f}")
        print(f"  Min/Max: {stats['min_delay']:.2f} / {stats['max_delay']:.2f}")
        print(f"\nPerformance:")
        print(f"  Mean Episode Length: {stats['mean_length']:.1f}")
        print(f"  Mean Assignments: {stats['mean_assignments']:.1f}")
        print(f"  Mean Completed Aircraft: {stats['mean_completed']:.1f}")
        print("="*60)
    
    def plot_results(self, save_path: str = None):
        """
        Create visualization plots
        
        Args:
            save_path: Path to save plots (if None, display only)
        """
        if not self.results:
            print("❌ No results to plot")
            return
        
        df = pd.DataFrame(self.results)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('RL Agent Evaluation Results', fontsize=16, fontweight='bold')
        
        # 1. Rewards over episodes
        axes[0, 0].plot(df['episode'], df['reward'], alpha=0.7, color='blue')
        axes[0, 0].axhline(df['reward'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Delay over episodes
        axes[0, 1].plot(df['episode'], df['delay'], alpha=0.7, color='orange')
        axes[0, 1].axhline(df['delay'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Total Delay')
        axes[0, 1].set_title('Episode Delays')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Episode length
        axes[0, 2].plot(df['episode'], df['length'], alpha=0.7, color='green')
        axes[0, 2].axhline(df['length'].mean(), color='red', linestyle='--', label='Mean')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Episode Length')
        axes[0, 2].set_title('Episode Lengths')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Reward distribution
        axes[1, 0].hist(df['reward'], bins=30, color='blue', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(df['reward'].mean(), color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Total Reward')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Reward Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Delay distribution
        axes[1, 1].hist(df['delay'], bins=30, color='orange', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(df['delay'].mean(), color='red', linestyle='--', linewidth=2)
        axes[1, 1].set_xlabel('Total Delay')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Delay Distribution')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Scatter: Reward vs Delay
        axes[1, 2].scatter(df['delay'], df['reward'], alpha=0.6, color='purple')
        axes[1, 2].set_xlabel('Total Delay')
        axes[1, 2].set_ylabel('Total Reward')
        axes[1, 2].set_title('Reward vs Delay')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plots saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def compare_with_baseline(self, baseline_results: List[Dict]):
        """
        Compare RL agent with baseline (e.g., random or heuristic)
        
        Args:
            baseline_results: Results from baseline policy
        """
        if not self.results or not baseline_results:
            print("❌ Need both RL and baseline results")
            return
        
        rl_df = pd.DataFrame(self.results)
        baseline_df = pd.DataFrame(baseline_results)
        
        print("\n" + "="*60)
        print("RL vs BASELINE COMPARISON")
        print("="*60)
        
        metrics = ['reward', 'delay', 'length', 'assignments']
        
        for metric in metrics:
            if metric in rl_df.columns and metric in baseline_df.columns:
                rl_mean = rl_df[metric].mean()
                baseline_mean = baseline_df[metric].mean()
                improvement = ((rl_mean - baseline_mean) / abs(baseline_mean) * 100)
                
                print(f"\n{metric.upper()}:")
                print(f"  RL: {rl_mean:.2f}")
                print(f"  Baseline: {baseline_mean:.2f}")
                print(f"  Improvement: {improvement:+.1f}%")
        
        print("="*60)
    
    def save_results(self, path: str):
        """Save results to CSV"""
        
        if not self.results:
            print("❌ No results to save")
            return
        
        df = pd.DataFrame(self.results)
        df.to_csv(path, index=False)
        print(f"✅ Results saved to {path}")