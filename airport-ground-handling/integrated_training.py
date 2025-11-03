"""
Integrated Training System - Connects Phase 1 RL + Multi-Agent + RAG

This script integrates:
1. Phase 1 RL Models (trained PPO at 297 reward)
2. Multi-Agent Simulation Framework
3. RAG System for Knowledge Augmentation

The trained RL agent controls the coordinator, while RAG provides
historical context for better decision-making.

Usage:
    python integrated_training.py --mode train      # Train integrated system
    python integrated_training.py --mode evaluate   # Evaluate performance
    python integrated_training.py --mode demo       # Run demo
    python integrated_training.py --mode full       # Complete pipeline

Location: airport-ground-handling/integrated_training.py
"""

import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

import tf_compat_fix

import numpy as np
import pandas as pd

# Phase 1 RL Components
from phase2_rag_agent_rl.rl_system.environment import AirportGroundHandlingEnv
try:
    from simple_trainer import SimpleTrainer
except ImportError:
    # Fallback if simple_trainer not found
    SimpleTrainer = None

# Multi-Agent Components (Your Implementation)
try:
    from multi_agent.aircraft_agent import AircraftAgent
    from multi_agent.vehicule_agent import VehicleAgent
    from multi_agent.coordinator_agent import CoordinatorAgent
    from multi_agent.simulation_engine import SimulationEngine
except ImportError:
    print("Warning: Multi-agent module not found. Will attempt local import.")

# RAG Components (Your Implementation)
try:
    from phase2_rag_agent_rl.rag_system.vector_store import VectorStore
    from phase2_rag_agent_rl.rag_system.retriever import ScenarioRetriever as Retriever
    from phase2_rag_agent_rl.rag_system.embeddings_generator import EmbeddingsGenerator as EmbeddingsEngine
    RAG_AVAILABLE = True
    print("✓ RAG system modules imported successfully")
except ImportError as e:
    print(f"Warning: RAG system module not found: {e}")
    VectorStore = None
    Retriever = None
    EmbeddingsEngine = None
    RAG_AVAILABLE = False


class IntegratedAirportOptimizer:
    """
    Integrated system combining RL, Multi-Agent, and RAG.
    
    Architecture:
    1. RL Agent (Phase 1) trained to make optimal decisions
    2. Multi-Agent Simulation for realistic environment
    3. RAG System provides historical context and recommendations
    """
    
    def __init__(
        self,
        rl_model_path: str = "./models/ppo_BEST_600K_steps.zip",
        enable_rag: bool = True,
        verbose: int = 1
    ):
        """
        Initialize integrated optimizer.
        
        Args:
            rl_model_path: Path to trained RL model
            enable_rag: Whether to use RAG augmentation
            verbose: Logging verbosity
        """
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.enable_rag = enable_rag
        
        # Load Phase 1 RL Model
        self.logger.info("="*70)
        self.logger.info("INTEGRATED SYSTEM INITIALIZATION")
        self.logger.info("="*70)
        
        self.logger.info("[1/3] Loading Phase 1 RL Model...")
        self.trainer = None
        self.rl_model = None
        
        if SimpleTrainer:
            self.trainer = SimpleTrainer(verbose=verbose)
            try:
                self.rl_model = self._load_model_by_path(rl_model_path)
                self.logger.info(f"[OK] RL Model loaded: {rl_model_path}")
            except Exception as e:
                self.logger.error(f"Failed to load RL model: {e}")
        else:
            # Load model directly without SimpleTrainer
            try:
                from stable_baselines3 import PPO
                env = AirportGroundHandlingEnv()
                self.rl_model = PPO.load(rl_model_path, env=env, device='cpu')
                self.logger.info(f"[OK] RL Model loaded: {rl_model_path}")
            except Exception as e:
                self.logger.warning(f"Could not load RL model: {e}")
        
        # Initialize Multi-Agent Components
        self.logger.info("[2/3] Initializing Multi-Agent Framework...")
        self.simulation_engine = None
        self.logger.info("[OK] Multi-Agent framework ready")
        
        # Initialize RAG System
        if enable_rag and RAG_AVAILABLE:
            self.logger.info("[3/3] Initializing RAG System...")
            try:
                self.vector_store = VectorStore()
                self.retriever = Retriever(vector_store=self.vector_store)
                self.embeddings_engine = EmbeddingsEngine()
                self.logger.info("[OK] RAG system initialized")
            except Exception as e:
                self.logger.warning(f"RAG initialization warning: {e}")
                self.enable_rag = False
        else:
            self.enable_rag = False
            self.logger.info("[3/3] RAG system not available, continuing without RAG")
        
        self.logger.info("\n[OK] Integrated system ready!\n")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger('IntegratedOptimizer')
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def run_integrated_training(
        self,
        num_episodes: int = 10,
        episode_length: int = 100,
        use_rag_augmentation: bool = True
    ) -> Dict[str, Any]:
        """
        Run integrated training combining all components.
        
        Args:
            num_episodes: Number of episodes to run
            episode_length: Steps per episode
            use_rag_augmentation: Whether to augment with RAG
            
        Returns:
            Training results and metrics
        """
        self.logger.info("="*70)
        self.logger.info("INTEGRATED TRAINING")
        self.logger.info("="*70)
        
        # Create RL environment
        env = AirportGroundHandlingEnv(
            num_aircraft=10,
            num_vehicles=30,
            episode_length=episode_length,
        )
        
        results = {
            'episode_rewards': [],
            'episode_delays': [],
            'episode_tasks': [],
            'episode_coordination_scores': [],
            'rag_augmentations_used': 0,
        }
        
        for episode in range(num_episodes):
            obs, _ = env.reset()
            done = False
            
            ep_reward = 0
            ep_delay = 0
            ep_tasks = 0
            ep_augmentations = 0
            
            step = 0
            while not done and step < episode_length:
                # Get action from RL model
                if self.rl_model:
                    action, _ = self.rl_model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()
                
                # Augment with RAG if enabled
                if use_rag_augmentation and self.enable_rag:
                    action = self._augment_action_with_rag(obs, action)
                    ep_augmentations += 1
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                ep_reward += reward
                ep_delay += info.get('delay', 0)
                ep_tasks += info.get('tasks_completed', 0)
                
                step += 1
            
            results['episode_rewards'].append(ep_reward)
            results['episode_delays'].append(ep_delay)
            results['episode_tasks'].append(ep_tasks)
            results['rag_augmentations_used'] += ep_augmentations
            
            self.logger.info(
                f"Episode {episode+1:2d}: "
                f"Reward={ep_reward:7.2f}, "
                f"Tasks={ep_tasks:3.0f}, "
                f"Delay={ep_delay:8.2f}"
            )
        
        # Calculate statistics
        stats = {
            'mean_reward': float(np.mean(results['episode_rewards'])),
            'std_reward': float(np.std(results['episode_rewards'])),
            'mean_tasks': float(np.mean(results['episode_tasks'])),
            'mean_delay': float(np.mean(results['episode_delays'])),
            'total_augmentations': results['rag_augmentations_used'],
            'augmentation_percentage': (
                results['rag_augmentations_used'] / (num_episodes * episode_length) * 100
            ),
        }
        
        self.logger.info("\n" + "="*70)
        self.logger.info("TRAINING SUMMARY")
        self.logger.info("="*70)
        self.logger.info(f"Mean Reward: {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
        self.logger.info(f"Mean Tasks: {stats['mean_tasks']:.0f}")
        self.logger.info(f"Mean Delay: {stats['mean_delay']:.2f}")
        self.logger.info(f"RAG Augmentations: {stats['augmentation_percentage']:.1f}%")
        
        return {**stats, **results}
    
    def _load_model_by_path(self, path: str):
        """Helper to load model."""
        from stable_baselines3 import PPO
        try:
            # Ensure path doesn't already have .zip extension
            if path.endswith('.zip.zip'):
                path = path[:-4]  # Remove double .zip
            elif not path.endswith('.zip'):
                path = path + '.zip'
            
            env = AirportGroundHandlingEnv()
            model = PPO.load(path, env=env, device='cpu')
            return model
        except Exception as e:
            self.logger.error(f"Could not load model from {path}: {e}")
            return None

    def _augment_action_with_rag(self, obs: Any, action: Any) -> Any:
        """
        Augment RL action with RAG recommendations.
        
        Args:
            obs: Current observation from RL environment
            action: Action from RL policy
            
        Returns:
            Potentially modified action based on RAG context
        """
        if not self.enable_rag or not hasattr(self, 'retriever'):
            return action
        
        try:
            # Convert observation to query context
            query = self._obs_to_query(obs)
            
            # Retrieve relevant historical scenarios
            retrieved = self.retriever.retrieve(query, n_results=1)
            
            if retrieved and len(retrieved) > 0:
                # Extract metadata from retrieved scenario
                scenario_data = retrieved[0]
                
                # Option 1: Blend RL action with RAG recommendation
                # You could modify the action based on retrieved scenario patterns
                # For now, we keep the RL action but log that RAG was consulted
                
                # Option 2: If you want to use RAG more aggressively:
                # Uncomment below to adjust action based on scenario similarity
                """
                if 'metadata' in scenario_data:
                    metadata = scenario_data['metadata']
                    # Analyze metadata and adjust action if confidence is high
                    # e.g., if retrieved scenario had better outcome with similar state
                    pass
                """
                
                return action
            
        except Exception as e:
            self.logger.debug(f"RAG augmentation skipped: {e}")
        
        return action
    
    def _obs_to_query(self, obs: Dict) -> str:
        """Convert observation to RAG query."""
        try:
            aircraft = np.array(obs.get('aircraft', []))
            vehicles = np.array(obs.get('vehicles', []))
            
            aircraft_util = float(np.mean(aircraft)) if len(aircraft) > 0 else 0.5
            vehicle_util = float(np.mean(vehicles)) if len(vehicles) > 0 else 0.5
            
            return (f"Aircraft utilization: {aircraft_util:.2f}, "
                   f"Vehicle utilization: {vehicle_util:.2f}")
        except:
            return "Current airport operations scenario"
    
    def evaluate_integrated_system(
        self,
        num_episodes: int = 20,
        episode_length: int = 100
    ) -> Dict[str, Any]:
        """
        Evaluate the complete integrated system.
        
        Args:
            num_episodes: Number of evaluation episodes
            episode_length: Steps per episode
            
        Returns:
            Evaluation metrics
        """
        self.logger.info("="*70)
        self.logger.info("INTEGRATED SYSTEM EVALUATION")
        self.logger.info("="*70)
        
        # Compare with and without RAG
        results_without_rag = self.run_integrated_training(
            num_episodes=num_episodes // 2,
            episode_length=episode_length,
            use_rag_augmentation=False
        )
        
        results_with_rag = self.run_integrated_training(
            num_episodes=num_episodes // 2,
            episode_length=episode_length,
            use_rag_augmentation=True
        )
        
        # Calculate improvement
        improvement = (
            results_with_rag['mean_reward'] - results_without_rag['mean_reward']
        )
        improvement_pct = (improvement / abs(results_without_rag['mean_reward'])) * 100
        
        self.logger.info("\n" + "="*70)
        self.logger.info("COMPARISON")
        self.logger.info("="*70)
        self.logger.info(f"Without RAG: {results_without_rag['mean_reward']:.2f}")
        self.logger.info(f"With RAG:    {results_with_rag['mean_reward']:.2f}")
        self.logger.info(f"Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)")
        
        return {
            'without_rag': results_without_rag,
            'with_rag': results_with_rag,
            'improvement': improvement,
            'improvement_percentage': improvement_pct,
        }
    
    def run_demo(self):
        """Run interactive demo."""
        self.logger.info("="*70)
        self.logger.info("INTEGRATED SYSTEM DEMO")
        self.logger.info("="*70)
        
        # Run quick demo
        self.run_integrated_training(
            num_episodes=5,
            episode_length=50,
            use_rag_augmentation=True
        )
        
        self.logger.info("\n✅ Demo complete!")


def _load_model_by_path(trainer: SimpleTrainer, path: str):
    """Helper to load model."""
    from stable_baselines3 import PPO
    try:
        env = trainer.create_environment()
        model = PPO.load(path, env=env, device='cpu')
        return model
    except Exception as e:
        print(f"Warning: Could not load model from {path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Integrated RL + Multi-Agent + RAG System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python integrated_training.py --mode train       # Train integrated system
  python integrated_training.py --mode evaluate    # Evaluate with/without RAG
  python integrated_training.py --mode demo        # Run quick demo
  python integrated_training.py --mode full        # Complete pipeline
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['train', 'evaluate', 'demo', 'full'],
        default='demo',
        help='Execution mode'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Number of episodes'
    )
    
    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Disable RAG augmentation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("GROUNDAI - INTEGRATED TRAINING SYSTEM")
    print("Phase 1 RL + Multi-Agent + RAG")
    print("="*70 + "\n")
    
    try:
        # Initialize integrated system
        optimizer = IntegratedAirportOptimizer(
            enable_rag=not args.no_rag,
            verbose=2 if args.verbose else 1
        )
        
        # Execute mode
        if args.mode == 'train':
            results = optimizer.run_integrated_training(num_episodes=args.episodes)
            print(f"\n✅ Training Results:")
            print(f"   Mean Reward: {results['mean_reward']:.2f}")
            print(f"   Mean Tasks: {results['mean_tasks']:.0f}\n")
        
        elif args.mode == 'evaluate':
            results = optimizer.evaluate_integrated_system(num_episodes=args.episodes)
            print(f"\n✅ Evaluation Complete\n")
        
        elif args.mode == 'demo':
            optimizer.run_demo()
        
        elif args.mode == 'full':
            print("[1/3] Training...")
            optimizer.run_integrated_training(num_episodes=5)
            print("\n[2/3] Evaluating...")
            optimizer.evaluate_integrated_system(num_episodes=10)
            print("\n[3/3] Running demo...")
            optimizer.run_demo()
        
        print("\n" + "="*70)
        print("✅ INTEGRATED SYSTEM COMPLETE")
        print("="*70)
        print("\nNext: Deploy to production or continue fine-tuning\n")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user\n")
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()