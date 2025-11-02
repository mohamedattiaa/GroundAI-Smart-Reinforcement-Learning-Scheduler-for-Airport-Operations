"""
Extended Training Script for GroundAI
Trains models for 500,000+ timesteps with checkpoints and detailed logging.

Usage:
    python train_extended.py --algo PPO --timesteps 500000
    python train_extended.py --algo DQN --timesteps 300000
    python train_extended.py --resume ./models/ppo_model_*.zip --additional 100000
"""

import argparse
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

import tf_compat_fix  # TensorFlow compatibility

from advanced_trainer import AdvancedTrainer


def main():
    parser = argparse.ArgumentParser(
        description='Extended Training for GroundAI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train PPO for 500,000 steps
  python train_extended.py --algo PPO --timesteps 500000
  
  # Train DQN for 300,000 steps
  python train_extended.py --algo DQN --timesteps 300000
  
  # Resume training from checkpoint
  python train_extended.py --resume ./models/ppo_model_20251101_164754.zip --additional 100000
  
  # Evaluate model
  python train_extended.py --evaluate ./models/ppo_model_20251101_164754.zip
        """
    )
    
    parser.add_argument(
        '--algo',
        choices=['PPO', 'A2C'],
        default='PPO',
        help='RL algorithm to train (Note: DQN requires Discrete actions, not MultiDiscrete)'
    )
    
    parser.add_argument(
        '--timesteps',
        type=int,
        default=500000,
        help='Total training timesteps'
    )
    
    parser.add_argument(
        '--num-envs',
        type=int,
        default=1,
        help='Number of parallel environments'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate (overrides default)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (for PPO/A2C)'
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Resume training from checkpoint'
    )
    
    parser.add_argument(
        '--additional',
        type=int,
        default=100000,
        help='Additional timesteps when resuming'
    )
    
    parser.add_argument(
        '--evaluate',
        type=str,
        default=None,
        help='Evaluate model and exit'
    )
    
    parser.add_argument(
        '--eval-episodes',
        type=int,
        default=20,
        help='Number of evaluation episodes'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("GROUNDAI EXTENDED TRAINING SYSTEM")
    print("="*70 + "\n")
    
    trainer = AdvancedTrainer(verbose=2 if args.verbose else 1)
    
    try:
        # Evaluation mode
        if args.evaluate:
            print(f"\n📊 Evaluating model: {args.evaluate}\n")
            stats = trainer.evaluate_detailed(
                args.evaluate,
                num_episodes=args.eval_episodes,
                save_stats=True
            )
            print("\n✅ Evaluation complete!\n")
            return 0
        
        # Resume training mode
        if args.resume:
            print(f"\n🔄 Resuming training from: {args.resume}\n")
            model_path = trainer.continuous_learning(
                args.resume,
                additional_timesteps=args.additional
            )
            print(f"\n✅ Extended training complete!")
            print(f"   Model saved: {model_path}\n")
            return 0
        
        # New training mode
        print(f"\n🎓 Starting {args.algo} training\n")
        
        # Prepare hyperparameters
        kwargs = {}
        if args.lr:
            kwargs['learning_rate'] = args.lr
        if args.batch_size and args.algo in ['PPO', 'A2C']:
            kwargs['batch_size'] = args.batch_size
        
        model_path = trainer.train_extended(
            total_timesteps=args.timesteps,
            algorithm=args.algo,
            num_envs=args.num_envs,
            **kwargs
        )
        
        print(f"\n✅ Training complete!")
        print(f"   Model saved: {model_path}")
        print(f"   Timesteps: {args.timesteps:,}")
        print(f"   Algorithm: {args.algo}\n")
        
        # Offer evaluation
        print("🧪 Would you like to evaluate the trained model?")
        print(f"   python train_extended.py --evaluate {model_path}\n")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user\n")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())