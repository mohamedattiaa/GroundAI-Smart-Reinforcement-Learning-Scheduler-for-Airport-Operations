#!/usr/bin/env python3
"""
Demo script for RL System

This demonstrates:
1. Creating RL environment
2. Training RL agent
3. Evaluating performance
4. Comparing with baseline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from phase2_rag_agent_rl.utils.data_loader import DatasetLoader
from phase2_rag_agent_rl.rl_system.environment import AirportEnv
from phase2_rag_agent_rl.rl_system.trainer import RLTrainer
from phase2_rag_agent_rl.rl_system.evaluator import RLEvaluator


def create_simple_scenario():
    """Create simple scenario for RL training"""
    
    return {
        'scenario_id': 'rl_train_001',
        'num_flights': 5,
        'flights': [
            {
                'flight_id': f'FL{i:03d}',
                'aircraft_type': np.random.choice(['A320', 'B737', 'B777']),
                'position': f'gate_{i % 3 + 1}',
                'actual_arrival': '2024-01-01 10:00:00',
                'scheduled_departure': '2024-01-01 11:00:00'
            }
            for i in range(5)
        ],
        'tasks': [],
        'statistics': {'total_tasks': 0, 'avg_delay': 0, 'equipment_failures': 0}
    }


def demo_environment():
    """Demo: Create and interact with environment"""
    
    print("="*70)
    print("DEMO 1: RL Environment")
    print("="*70)
    
    # Create scenario
    scenario = create_simple_scenario()
    
    # Add tasks
    for flight in scenario['flights']:
        for task_name in ['deplaning', 'refueling', 'boarding']:
            scenario['tasks'].append({
                'flight_id': flight['flight_id'],
                'task_name': task_name,
                'duration': 10,
                'required_vehicles': ['fuel_truck'] if task_name == 'refueling' else ['passenger_stairs'],
                'predecessors': []
            })
    
    scenario['statistics']['total_tasks'] = len(scenario['tasks'])
    
    # Vehicle config
    vehicle_config = {
        'fuel_truck': {'count': 2, 'max_tasks_before_base': 10, 'compatible_aircraft': ['all']},
        'passenger_stairs': {'count': 2, 'max_tasks_before_base': 15, 'compatible_aircraft': ['all']}
    }
    
    # Create environment
    print("\n1️⃣  Creating environment...")
    env = AirportEnv(scenario, vehicle_config, max_steps=50)
    
    print(f"✅ Environment created")
    print(f"   Observation space: {env.observation_space}")
    print(f"   Action space: {env.action_space}")
    
    # Test random actions
    print("\n2️⃣  Testing random actions...")
    
    obs, info = env.reset()
    total_reward = 0
    
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        if step < 3:
            print(f"   Step {step + 1}: Reward={reward:.2f}, Delay={info['total_delay']:.2f}")
        
        if terminated or truncated:
            break
    
    print(f"\n✅ Random policy total reward: {total_reward:.2f}")
    print(f"   Total delay: {info['total_delay']:.2f}")
    
    input("\n\nPress Enter to continue...")


def demo_training():
    """Demo: Train RL agent"""
    
    print("\n" + "="*70)
    print("DEMO 2: Training RL Agent")
    print("="*70)
    
    # Create scenario
    scenario = create_simple_scenario()
    
    # Add tasks
    for flight in scenario['flights']:
        for task_name in ['deplaning', 'refueling', 'boarding']:
            scenario['tasks'].append({
                'flight_id': flight['flight_id'],
                'task_name': task_name,
                'duration': 10,
                'required_vehicles': ['fuel_truck'] if task_name == 'refueling' else ['passenger_stairs'],
                'predecessors': []
            })
    
    scenario['statistics']['total_tasks'] = len(scenario['tasks'])
    
    vehicle_config = {
        'fuel_truck': {'count': 2, 'max_tasks_before_base': 10, 'compatible_aircraft': ['all']},
        'passenger_stairs': {'count': 2, 'max_tasks_before_base': 15, 'compatible_aircraft': ['all']}
    }
    
    # Create environment
    print("\n1️⃣  Creating environment...")
    env = AirportEnv(scenario, vehicle_config, max_steps=50)
    
    # Create trainer
    print("\n2️⃣  Creating trainer...")
    trainer = RLTrainer(env, algorithm='PPO', device='cpu')
    
    # Train (short demo)
    print("\n3️⃣  Training agent (quick demo)...")
    print("   Note: This is a short demo. Real training needs 100k+ steps")
    
    trainer.train(
        total_timesteps=5000,  # Short for demo
        save_freq=2000,
        save_path='models/rl_checkpoints'
    )
    
    # Evaluate
    print("\n4️⃣  Evaluating trained agent...")
    results = trainer.evaluate(n_episodes=5)
    
    print(f"\n✅ Training demo complete")
    print(f"   Mean reward: {results['mean_reward']:.2f}")
    print(f"   Mean delay: {results['mean_delay']:.2f}")
    
    input("\n\nPress Enter to continue...")


def demo_evaluation():
    """Demo: Evaluate and visualize results"""
    
    print("\n" + "="*70)
    print("DEMO 3: Evaluation & Visualization")
    print("="*70)
    
    # Create scenario
    scenario = create_simple_scenario()
    
    for flight in scenario['flights']:
        for task_name in ['deplaning', 'refueling', 'boarding']:
            scenario['tasks'].append({
                'flight_id': flight['flight_id'],
                'task_name': task_name,
                'duration': 10,
                'required_vehicles': ['fuel_truck'] if task_name == 'refueling' else ['passenger_stairs'],
                'predecessors': []
            })
    
    scenario['statistics']['total_tasks'] = len(scenario['tasks'])
    
    vehicle_config = {
        'fuel_truck': {'count': 2, 'max_tasks_before_base': 10, 'compatible_aircraft': ['all']},
        'passenger_stairs': {'count': 2, 'max_tasks_before_base': 15, 'compatible_aircraft': ['all']}
    }
    
    # Create environment and trainer
    env = AirportEnv(scenario, vehicle_config, max_steps=50)
    trainer = RLTrainer(env, algorithm='PPO', device='cpu')
    
    # Quick training
    print("\n1️⃣  Quick training...")
    trainer.train(total_timesteps=2000, save_path='models/rl_checkpoints')
    
    # Create evaluator
    print("\n2️⃣  Creating evaluator...")
    evaluator = RLEvaluator(env, trainer.model)
    
    # Run evaluation episodes
    print("\n3️⃣  Running evaluation episodes...")
    evaluator.run_episodes(n_episodes=20)
    
    # Print statistics
    evaluator.print_statistics()
    
    # Create plots
    print("\n4️⃣  Creating visualizations...")
    evaluator.plot_results(save_path='data/statistics/rl_evaluation.png')
    
    # Save results
    evaluator.save_results('data/statistics/rl_results.csv')
    
    print("\n✅ Evaluation complete")


def demo_comparison():
    """Demo: Compare RL with baseline"""
    
    print("\n" + "="*70)
    print("DEMO 4: RL vs Baseline Comparison")
    print("="*70)
    
    # Create scenario
    scenario = create_simple_scenario()
    
    for flight in scenario['flights']:
        for task_name in ['deplaning', 'refueling', 'boarding']:
            scenario['tasks'].append({
                'flight_id': flight['flight_id'],
                'task_name': task_name,
                'duration': 10,
                'required_vehicles': ['fuel_truck'] if task_name == 'refueling' else ['passenger_stairs'],
                'predecessors': []
            })
    
    scenario['statistics']['total_tasks'] = len(scenario['tasks'])
    
    vehicle_config = {
        'fuel_truck': {'count': 2, 'max_tasks_before_base': 10, 'compatible_aircraft': ['all']},
        'passenger_stairs': {'count': 2, 'max_tasks_before_base': 15, 'compatible_aircraft': ['all']}
    }
    
    env = AirportEnv(scenario, vehicle_config, max_steps=50)
    
    # 1. Baseline (random policy)
    print("\n1️⃣  Evaluating baseline (random policy)...")
    
    baseline_results = []
    for ep in range(20):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        baseline_results.append({
            'episode': ep,
            'reward': episode_reward,
            'delay': info['total_delay'],
            'length': info['step'],
            'assignments': info['assignments']
        })
    
    baseline_mean = np.mean([r['reward'] for r in baseline_results])
    print(f"   Baseline mean reward: {baseline_mean:.2f}")
    
    # 2. RL policy
    print("\n2️⃣  Training and evaluating RL agent...")
    
    trainer = RLTrainer(env, algorithm='PPO', device='cpu')
    trainer.train(total_timesteps=5000, save_path='models/rl_checkpoints')
    
    evaluator = RLEvaluator(env, trainer.model)
    evaluator.run_episodes(n_episodes=20)
    
    # 3. Compare
    print("\n3️⃣  Comparing results...")
    evaluator.compare_with_baseline(baseline_results)
    
    print("\n✅ Comparison complete")


def main():
    """Main demo function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="RL System Demo")
    parser.add_argument(
        '--mode',
        choices=['environment', 'training', 'evaluation', 'comparison', 'all'],
        default='all',
        help='Demo mode to run'
    )
    
    args = parser.parse_args()
    
    print("\n")
    print("="*70)
    print("REINFORCEMENT LEARNING SYSTEM DEMO")
    print("="*70)
    
    if args.mode in ['environment', 'all']:
        demo_environment()
    
    if args.mode in ['training', 'all']:
        demo_training()
    
    if args.mode in ['evaluation', 'all']:
        demo_evaluation()
    
    if args.mode in ['comparison', 'all']:
        demo_comparison()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("\nTo run specific demos:")
    print("  python demo_rl.py --mode environment")
    print("  python demo_rl.py --mode training")
    print("  python demo_rl.py --mode evaluation")
    print("  python demo_rl.py --mode comparison")


if __name__ == "__main__":
    main()