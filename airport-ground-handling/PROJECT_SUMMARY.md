# GroundAI: Smart Reinforcement Learning Scheduler for Airport Operations

## 🎯 Project Summary

GroundAI is an advanced Reinforcement Learning system that optimizes airport ground handling operations. It coordinates multiple agents to schedule and execute ground services (fueling, catering, cleaning) efficiently while minimizing delays and maximizing resource utilization.

---

## 📊 Current Status

### ✅ Completed Components

1. **Environment** (`environment.py`)
   - Multi-agent Airport Ground Handling Gym environment
   - Dict observation space (aircraft, global, vehicles)
   - Multi-discrete action space
   - Proper reward shaping
   - Delay and task tracking

2. **Policies** (`policies.py`)
   - CustomFeaturesExtractor for Dict observation spaces
   - GroundHandlingActorCriticPolicy (PPO/A2C)
   - GroundHandlingDQNPolicy
   - MultiAgentPolicyNetwork
   - DuelingDQNNetwork
   - TaskPriorityNetwork

3. **Training System** (`trainer.py` & `simple_trainer.py`)
   - Full RLTrainer with YAML config support
   - SimpleTrainer for quick testing
   - PPO, DQN, A2C algorithm support
   - Checkpoint and evaluation callbacks
   - Comprehensive logging

4. **Demo & Testing** (`demo_rl.py`)
   - Environment demonstration
   - Training pipeline (✅ Working!)
   - Evaluation framework
   - Algorithm comparison
   - Random policy baseline

5. **Dashboard** (`dashboard.py`)
   - Real-time training visualization
   - Model evaluation interface
   - Environment configuration viewer
   - Metrics and statistics

### 📈 Training Results (Latest)

```
Training Timesteps: 50,000
Algorithm: PPO
Device: CPU
Duration: 3 min 16 sec

Performance Progression:
- Start: -188.27 ± 154.72 (random behavior)
- Mid:   -32.79 ± 97.79   (improving)
- End:   +38.61 ± 275.87  (learned policy)

Improvement: +226.88 points (120% increase)

Model Path: ./models/ppo_model_20251101_164754
```

---

## 🚀 Quick Start Guide

### 1. Setup Environment

```bash
# Create conda environment
conda create --name py39_env python=3.9 -y
conda activate py39_env

# Install PyTorch
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

# Install requirements
pip install -r requirements.txt
```

### 2. Run Demo

```bash
# Test environment
python demo_rl.py --mode environment

# Train a new model (50,000 timesteps)
python demo_rl.py --mode training

# Evaluate trained model
python demo_rl.py --mode evaluation

# Compare algorithms
python demo_rl.py --mode comparison

# Run all demos
python demo_rl.py --mode all
```

### 3. Launch Dashboard

```bash
streamlit run dashboard.py
```

Then open http://localhost:8501 in your browser.

### 4. Custom Training

```python
from simple_trainer import SimpleTrainer

trainer = SimpleTrainer(verbose=1)

# Train PPO
model_path = trainer.train_ppo(
    total_timesteps=100000,
    num_envs=1,
    learning_rate=3e-4,
    batch_size=64
)

# Evaluate
stats = trainer.evaluate(model_path, num_episodes=10)
print(f"Mean Reward: {stats['mean_reward']:.2f}")
```

---

## 📁 Project Structure

```
airport-ground-handling/
│
├── phase2_rag_agent_rl/
│   ├── rl_system/
│   │   ├── __init__.py
│   │   ├── environment.py           # Main Gym environment
│   │   ├── policies.py              # Custom RL policies
│   │   ├── trainer.py               # Full trainer with YAML
│   │   └── simple_trainer.py        # Simplified trainer
│   │
│   └── rag_system/
│       ├── __init__.py
│       └── rag_agent.py             # RAG components (future)
│
├── models/                          # Trained models directory
│   └── ppo_model_20251101_164754.zip
│
├── logs/                            # Training logs
│   ├── airport_rl.log
│   └── training.log
│
├── checkpoints/                     # Training checkpoints
├── videos/                          # Recorded episodes
│
├── config.yaml                      # Configuration file
├── demo_rl.py                       # Main demo script
├── dashboard.py                     # Streamlit dashboard
├── simple_trainer.py                # Standalone trainer
├── policies.py                      # (Alias for policies)
├── tf_compat_fix.py                 # TensorFlow compatibility
├── requirements.txt                 # Dependencies
├── README.md                        # Documentation
└── PROJECT_SUMMARY.md               # This file
```

---

## 🎮 Environment Details

### Observation Space
```python
{
    'aircraft': Box(0, 1, (10, 8))    # [id, arrival, tasks, priority, urgency, fueling, catering, cleaning]
    'global': Box(0, 1, (5,))          # [total_delay, avg_util, resources, progress, task_rate]
    'vehicles': Box(0, 1, (30, 6))    # [id, availability, task_type, progress, efficiency, idle_time]
}
```

### Action Space
```python
MultiDiscrete([10, 3, 30])
# [select_aircraft, select_task, select_vehicle]
# 10 aircraft × 3 task types × 30 vehicles
```

### Tasks
| Task | Duration | Priority | Resource |
|------|----------|----------|----------|
| Fueling | 5-15 min | High (1.0) | 1 vehicle |
| Catering | 10-25 min | Medium (0.8) | 1 vehicle |
| Cleaning | 15-30 min | Low (0.6) | 1 vehicle |

### Reward Structure
```python
{
    'task_completion': +50.0,           # Completing a task
    'delay_penalty_per_step': -1.0,     # Each timestep of delay
    'idle_penalty': -0.5,               # Idle resources
    'action_efficiency': +10.0,         # Valid actions
    'collision_penalty': -100.0,        # Resource conflicts
}
```

---

## 🤖 Algorithms

### PPO (Proximal Policy Optimization)
- **Status**: ✅ Working
- **Hyperparameters**:
  - Learning rate: 3.0e-4
  - N steps: 2048
  - Batch size: 64
  - Epochs: 10
  - Gamma: 0.99
  - GAE Lambda: 0.95
  - Clip range: 0.2

### DQN (Deep Q-Network)
- **Status**: ✅ Available
- **Hyperparameters**:
  - Learning rate: 1.0e-4
  - Buffer size: 100,000
  - Batch size: 32
  - Gamma: 0.99
  - Exploration fraction: 0.1

### A2C (Advantage Actor-Critic)
- **Status**: ✅ Available
- **Hyperparameters**:
  - Learning rate: 7.0e-4
  - N steps: 5
  - Gamma: 0.99
  - GAE Lambda: 0.95

---

## 📊 Training Metrics

### Key Indicators
- **Mean Episode Reward**: Cumulative reward per episode
- **Episode Length**: Steps to completion (fixed at 100)
- **Total Delay**: Sum of all scheduling delays
- **Tasks Completed**: Number of successfully completed tasks
- **Resource Utilization**: Percentage of active vehicles
- **Collision Rate**: Conflicts detected

### Loss Functions
- **Policy Gradient Loss**: Improving policy updates
- **Value Loss**: State value estimation error
- **Entropy Loss**: Exploration diversity

---

## 🔧 Configuration

### Via YAML (`config.yaml`)
```yaml
environment:
  num_aircraft: 10
  num_vehicles: 30
  episode_length: 100

training:
  algorithm: "PPO"
  total_timesteps: 100000
  eval_freq: 5000
  n_eval_episodes: 5

ppo:
  learning_rate: 3.0e-4
  n_steps: 2048
  batch_size: 64
```

### Via Python
```python
trainer = SimpleTrainer(verbose=1)
trainer.train_ppo(
    total_timesteps=50000,
    learning_rate=3e-4,
    batch_size=64
)
```

---

## 🧪 Evaluation

### Single Model Evaluation
```bash
python demo_rl.py --mode evaluation
```

### Custom Evaluation
```python
from simple_trainer import SimpleTrainer

trainer = SimpleTrainer()
stats = trainer.evaluate(
    model_path="./models/ppo_model_20251101_164754.zip",
    num_episodes=10
)

print(f"Mean Reward: {stats['mean_reward']:.2f}")
print(f"Std Dev: {stats['std_reward']:.2f}")
print(f"Mean Delay: {stats['mean_delay']:.2f}")
print(f"Avg Tasks: {stats['mean_tasks']:.2f}")
```

---

## 📈 Performance Benchmarks

### CPU (Current)
- Training speed: ~258 FPS
- 50,000 timesteps: ~3.2 minutes
- Memory usage: ~1.2 GB

### GPU (Recommended)
- Expected speedup: 5-10x
- Install: `pip install torch --cuda`

---

## 🐛 Troubleshooting

### ImportError: stable_baselines3
```bash
pip install --upgrade stable-baselines3
```

### Unicode Error (Windows)
Already fixed with UTF-8 encoding in logging

### Out of Memory
Reduce batch size or use fewer parallel environments:
```python
trainer.train_ppo(batch_size=32, num_envs=1)
```

### Model Not Found
```bash
# Check models directory
ls models/

# Train a new model
python demo_rl.py --mode training
```

---

## 🚀 Next Steps

### Short Term (Week 1-2)
- [ ] Extend training to 500,000 timesteps
- [ ] Compare PPO vs DQN vs A2C performance
- [ ] Implement multi-agent coordination
- [ ] Add visualization to dashboard

### Medium Term (Week 3-4)
- [ ] Integrate RAG system for decision support
- [ ] Test on larger environments (50+ aircraft)
- [ ] Implement curriculum learning
- [ ] Add real airport data

### Long Term (Month 2+)
- [ ] Deploy to production simulator
- [ ] Real-time optimization pipeline
- [ ] Advanced multi-agent learning
- [ ] Transfer learning across scenarios

---

## 📚 Resources

### Documentation
- [Stable Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Docs](https://gymnasium.farama.org/)
- [PyTorch Docs](https://pytorch.org/docs/)

### Papers
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [DQN Paper](https://arxiv.org/abs/1312.5602)
- [Actor-Critic Methods](https://arxiv.org/abs/1602.01783)

### Related Projects
- OpenAI Gym
- Stable Baselines3
- Ray RLlib

---

## 📝 License

MIT License - See LICENSE file for details

---

## 👥 Contributors

- Mohamed Attia (@mohamedattiaa)

---

## 📞 Support

For issues, questions, or suggestions:
1. Check README.md and documentation
2. Review logs in `./logs/`
3. Run demo with `--verbose` flag
4. Create GitHub issue

---

**Last Updated**: November 1, 2025  
**Project Status**: ✅ Training Phase Complete | 🚀 Ready for Evaluation