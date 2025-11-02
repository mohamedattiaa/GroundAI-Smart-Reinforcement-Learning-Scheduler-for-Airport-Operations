"""
Training configuration and defaults for RL system.
"""

from typing import Dict, Any


def get_ppo_config() -> Dict[str, Any]:
    """Get default PPO training configuration."""
    return {
        'learning_rate': 3.0e-4,
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
    }


def get_dqn_config() -> Dict[str, Any]:
    """Get default DQN training configuration."""
    return {
        'learning_rate': 1.0e-4,
        'buffer_size': 100000,
        'learning_starts': 1000,
        'batch_size': 32,
        'tau': 1.0,
        'gamma': 0.99,
        'exploration_fraction': 0.1,
        'exploration_final_eps': 0.05,
        'target_update_interval': 10000,
    }


def get_a2c_config() -> Dict[str, Any]:
    """Get default A2C training configuration."""
    return {
        'learning_rate': 7.0e-4,
        'n_steps': 5,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'rms_prop_eps': 1e-5,
    }


def get_environment_config() -> Dict[str, Any]:
    """Get default environment configuration."""
    return {
        'num_aircraft': 10,
        'num_vehicles': 30,
        'episode_length': 100,
        'num_episodes': 10,
        'reward_shaping': True,
        'normalize_obs': True,
        'normalize_reward': False,
    }


def get_training_defaults() -> Dict[str, Any]:
    """Get all training defaults."""
    return {
        'algorithm': 'PPO',
        'total_timesteps': 100000,
        'num_envs': 1,
        'eval_freq': 5000,
        'n_eval_episodes': 5,
        'save_freq': 10000,
        'checkpoint_dir': './checkpoints',
        'model_dir': './models',
        'log_dir': './logs',
        'seed': 42,
        'device': 'auto',
        'verbose': 1,
        'render': False,
        'early_stopping_patience': 20,
        'target_reward': 400.0,
    }


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Merge override config into base config."""
    result = base.copy()
    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result