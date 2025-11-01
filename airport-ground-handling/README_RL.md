# Reinforcement Learning System for Airport Ground Handling

## Overview

This RL system learns optimal vehicle assignment policies through trial and error, improving scheduling decisions over time.

## Architecture
```
Environment (AirportEnv)
    ↓
RL Agent (PPO/DQN/A2C)
    ↓
Actions (vehicle assignments)
    ↓
Rewards (based on delay reduction)
```

## Quick Start
```bash
# Run environment demo
python demo_rl.py --mode environment

# Train agent (quick demo)
python demo_rl.py --mode training

# Evaluate and visualize
python demo_rl.py --mode evaluation

# Compare with baseline
python demo_rl.py --mode comparison

# Run all demos
python demo_rl.py --mode all
```

## State Space

The agent observes:
- Aircraft states (completion %, delay, type)
- Vehicle states (availability, workload, position)
- Global state (time, total assignments, total delay)

## Action Space

Agent chooses:
- Which aircraft to serve
- Which task to assign
- Which vehicle to use

## Reward Function

- **+10** for completing a task
- **+20** bonus for completing all aircraft tasks
- **-0.5** per minute of delay
- **-10** for invalid actions

## Training
```python
from phase2_rag_agent_rl.rl_system import AirportEnv, RLTrainer

env = AirportEnv(scenario, vehicle_config)
trainer = RLTrainer(env, algorithm='PPO')
trainer.train(total_timesteps=100000)
```

## Evaluation
```python
from phase2_rag_agent_rl.rl_system import RLEvaluator

evaluator = RLEvaluator(env, model)
evaluator.run_episodes(n_episodes=100)
evaluator.print_statistics()
evaluator.plot_results()
```

## Algorithms Supported

- **PPO** (Proximal Policy Optimization) - Best for continuous learning
- **DQN** (Deep Q-Network) - Good for discrete actions
- **A2C** (Advantage Actor-Critic) - Fast training

## Performance Tips

- Start with PPO (most stable)
- Train for 100k+ timesteps
- Use GPU if available
- Monitor delay reduction metric
- Compare with baseline policies

## Results Interpretation

- **Negative rewards** initially are normal
- Look for **decreasing delay** over time
- **Completion rate** should increase
- Compare with **random/heuristic baselines**