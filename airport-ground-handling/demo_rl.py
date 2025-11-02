"""
Demo script for GroundAI Reinforcement Learning System.
Tests environment, training, evaluation, and algorithm comparison.
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# TensorFlow compatibility
import tf_compat_fix

import numpy as np
from tabulate import tabulate

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from phase2_rag_agent_rl.rl_system.environment import AirportGroundHandlingEnv
try:
    from simple_trainer import SimpleTrainer as RLTrainer
    USE_SIMPLE_TRAINER = True
except ImportError:
    from phase2_rag_agent_rl.rl_system.trainer import RLTrainer
    USE_SIMPLE_TRAINER = False


def demo_environment():
    """Demo 1: Test environment creation and basic interaction."""
    print("\n" + "="*70)
    print("DEMO 1: RL ENVIRONMENT")
    print("="*70)
    
    print("\n1️⃣  Creating environment...")
    env = AirportGroundHandlingEnv(
        num_aircraft=10,
        num_vehicles=30,
        episode_length=100,
    )
    print("✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    print("\n2️⃣  Testing random actions...")
    obs, _ = env.reset()
    
    rewards_list = []
    delays_list = []
    task_completions = []
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards_list.append(reward)
        delays_list.append(info.get('delay', 0))
        task_completions.append(info.get('tasks_completed', 0))
        
        print(f"   Step {step+1}: Reward={reward:7.2f}, Delay={info.get('delay', 0):6.2f}, "
              f"Tasks={info.get('tasks_completed', 0)}")
    
    total_reward = sum(rewards_list)
    total_delay = sum(delays_list)
    
    print(f"\n✅ Random policy summary:")
    print(f"   Total reward: {total_reward:.2f} (avg: {np.mean(rewards_list):.2f})")
    print(f"   Total delay: {total_delay:.2f}")
    print(f"   Tasks completed: {sum(task_completions)}")
    print(f"   Reward variance: {np.var(rewards_list):.2f}")
    
    env.close()


def demo_training():
    """Demo 2: Train an agent."""
    print("\n" + "="*70)
    print("DEMO 2: TRAINING")
    print("="*70)
    
    print("\n1️⃣  Initializing trainer...")
    trainer = RLTrainer(verbose=1)
    print("✅ Trainer initialized")
    
    print("\n2️⃣  Starting training (50,000 timesteps, PPO)...")
    print("   (This is a demo - first run may take 2-5 minutes)")
    try:
        model_path = trainer.train_ppo(
            total_timesteps=50000,
            num_envs=1,
            learning_rate=3e-4,
            batch_size=64
        )
        print(f"✅ Training completed")
        print(f"   Model saved: {model_path}")
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()


def demo_evaluation():
    """Demo 3: Evaluate a trained agent."""
    print("\n" + "="*70)
    print("DEMO 3: EVALUATION")
    print("="*70)
    
    # Check if model exists
    model_files = list(Path('./models').glob('ppo*.zip'))
    
    if not model_files:
        print("\n⚠️  No trained PPO model found.")
        print("   Train a model first with: python demo_rl.py --mode training")
        return
    
    latest_model = max(model_files, key=lambda p: p.stat().st_mtime)
    
    print(f"\n1️⃣  Loading model: {latest_model.name}")
    
    trainer = RLTrainer(verbose=1)
    
    print(f"\n2️⃣  Evaluating on 5 episodes...")
    try:
        stats = trainer.evaluate(str(latest_model), num_episodes=5)
        
        print("\n✅ Evaluation Results:")
        print(f"   Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        print(f"   Mean Delay: {stats['mean_delay']:.2f}")
        print(f"   Avg Tasks/Episode: {stats['mean_tasks']:.2f}")
    
    except Exception as e:
        print(f"❌ Evaluation error: {str(e)}")
        import traceback
        traceback.print_exc()


def demo_comparison():
    """Demo 4: Compare different algorithms."""
    print("\n" + "="*70)
    print("DEMO 4: ALGORITHM COMPARISON")
    print("="*70)
    
    print("\n1️⃣  Creating comparison environment...")
    env = AirportGroundHandlingEnv(
        num_aircraft=10,
        num_vehicles=30,
        episode_length=50,
    )
    
    print("\n2️⃣  Running random policy on 3 episodes...")
    
    comparison_data = []
    
    for episode in range(3):
        obs, _ = env.reset()
        
        total_reward = 0
        total_delay = 0
        total_tasks = 0
        
        for step in range(50):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            total_delay += info.get('delay', 0)
            total_tasks += info.get('tasks_completed', 0)
        
        comparison_data.append({
            'Algorithm': 'Random',
            'Episode': episode + 1,
            'Total Reward': f"{total_reward:.2f}",
            'Avg Delay': f"{total_delay/50:.2f}",
            'Tasks': int(total_tasks)
        })
    
    print("\n✅ Comparison Results:")
    print(tabulate(comparison_data, headers='keys', tablefmt='grid'))
    
    env.close()


def main():
    """Main demo runner."""
    parser = argparse.ArgumentParser(
        description='GroundAI RL System Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_rl.py --mode environment
  python demo_rl.py --mode training
  python demo_rl.py --mode evaluation
  python demo_rl.py --mode comparison
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['environment', 'training', 'evaluation', 'comparison', 'all'],
        default='all',
        help='Which demo to run'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("REINFORCEMENT LEARNING SYSTEM DEMO - GroundAI")
    print("="*70)
    
    try:
        if args.mode in ['environment', 'all']:
            demo_environment()
        
        if args.mode in ['training', 'all']:
            demo_training()
        
        if args.mode in ['evaluation', 'all']:
            demo_evaluation()
        
        if args.mode in ['comparison', 'all']:
            demo_comparison()
        
        print("\n" + "="*70)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY")
        print("="*70)
        
        print("\n📚 Next Steps:")
        print("  1. Review the config.yaml for customization")
        print("  2. Run: python demo_rl.py --mode training")
        print("  3. Check logs/ and models/ directories")
        print("  4. Try: streamlit run dashboard.py")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()