# Multi-Agent System for Airport Ground Handling

## Overview

This multi-agent system simulates coordinated operations between aircraft and ground service vehicles, enabling decentralized decision-making and resource allocation.

## Architecture
```
┌─────────────────┐
│ Aircraft Agents │  (Request services)
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Coordinator    │  (Matches resources)
│     Agent       │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ Vehicle Agents  │  (Bid on requests)
└─────────────────┘
```

## Agents

### 1. Aircraft Agent
- Manages turnaround for one aircraft
- Requests ground services
- Tracks task completion
- Calculates priority scores

### 2. Vehicle Agent
- Represents one ground service vehicle
- Bids on service requests
- Manages its schedule
- Tracks workload

### 3. Coordinator Agent
- Central coordinator
- Collects requests and bids
- Makes assignment decisions
- Resolves conflicts

## Quick Start
```bash
# Run basic demo
python demo_agents.py --mode basic

# Run full simulation
python demo_agents.py --mode simulation

# Run all demos
python demo_agents.py --mode all
```

## Usage Examples

### Example 1: Create Agents
```python
from phase2_rag_agent_rl.multi_agent import AircraftAgent, VehicleAgent

# Create aircraft agent
aircraft = AircraftAgent(
    flight_id='FL001',
    aircraft_type='A320',
    position='gate_5',
    arrival_time=pd.Timestamp.now(),
    scheduled_departure=pd.Timestamp.now() + pd.Timedelta(minutes=45),
    tasks=tasks
)

# Create vehicle agent
vehicle = VehicleAgent(
    vehicle_id='fuel_truck_1',
    vehicle_type='fuel_truck',
    max_tasks=10,
    compatible_aircraft=['all']
)
```

### Example 2: Run Simulation
```python
from phase2_rag_agent_rl.multi_agent import SimulationEngine

simulation = SimulationEngine(
    scenario=scenario,
    vehicle_config=vehicle_config,
    travel_times=travel_times
)

results = simulation.run(max_rounds=50)
```

## Configuration

Edit `configs/agent_config.yaml`:
```yaml
aircraft_agent:
  priority_weights:
    time_until_departure: 0.4
    aircraft_size: 0.3

vehicle_agent:
  bidding_weights:
    availability_time: 0.5
    current_workload: 0.3
```

## Key Features

- **Decentralized Decision Making**: Agents make autonomous decisions
- **Bidding System**: Vehicles compete for assignments
- **Priority-Based Scheduling**: Critical flights get priority
- **Conflict Resolution**: Coordinator resolves resource conflicts
- **Real-time Adaptation**: Agents respond to changing conditions

## Performance Metrics

- Total delay reduction
- Vehicle utilization rate
- Assignment efficiency
- Conflict resolution rate

## Testing
```bash
python test_agents.py
```

## Next Steps

1. Integrate with RAG system for intelligent recommendations
2. Add learning capabilities (RL agents)
3. Real-time visualization dashboard