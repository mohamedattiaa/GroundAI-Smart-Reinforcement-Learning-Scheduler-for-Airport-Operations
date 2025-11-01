#!/usr/bin/env python3
"""
Fixed Demo script for Multi-Agent System
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import json
from phase2_rag_agent_rl.utils.data_loader import DatasetLoader
from phase2_rag_agent_rl.multi_agent.aircraft_agent import AircraftAgent
from phase2_rag_agent_rl.multi_agent.vehicle_agent import VehicleAgent
from phase2_rag_agent_rl.multi_agent.coordinator_agent import CoordinatorAgent
from phase2_rag_agent_rl.multi_agent.simulation_engine import SimulationEngine


def create_sample_scenario():
    """Create a simple test scenario with known structure"""
    
    return {
        'scenario_id': 'test_001',
        'num_flights': 2,
        'flights': [
            {
                'flight_id': 'FL001',
                'aircraft_type': 'A320',
                'position': 'gate_1',
                'actual_arrival': pd.Timestamp.now().isoformat(),
                'scheduled_departure': (pd.Timestamp.now() + pd.Timedelta(minutes=45)).isoformat()
            },
            {
                'flight_id': 'FL002',
                'aircraft_type': 'B737',
                'position': 'gate_2',
                'actual_arrival': pd.Timestamp.now().isoformat(),
                'scheduled_departure': (pd.Timestamp.now() + pd.Timedelta(minutes=45)).isoformat()
            }
        ],
        'tasks': [
            # FL001 tasks
            {'flight_id': 'FL001', 'task_name': 'deplaning', 'duration': 7, 
             'required_vehicles': ['passenger_stairs'], 'predecessors': []},
            {'flight_id': 'FL001', 'task_name': 'refueling', 'duration': 12, 
             'required_vehicles': ['fuel_truck'], 'predecessors': ['deplaning']},
            {'flight_id': 'FL001', 'task_name': 'catering', 'duration': 15, 
             'required_vehicles': ['catering_truck'], 'predecessors': ['deplaning']},
            {'flight_id': 'FL001', 'task_name': 'boarding', 'duration': 15, 
             'required_vehicles': ['passenger_stairs'], 'predecessors': ['catering']},
            
            # FL002 tasks
            {'flight_id': 'FL002', 'task_name': 'deplaning', 'duration': 7, 
             'required_vehicles': ['passenger_stairs'], 'predecessors': []},
            {'flight_id': 'FL002', 'task_name': 'refueling', 'duration': 12, 
             'required_vehicles': ['fuel_truck'], 'predecessors': ['deplaning']},
            {'flight_id': 'FL002', 'task_name': 'catering', 'duration': 15, 
             'required_vehicles': ['catering_truck'], 'predecessors': ['deplaning']},
            {'flight_id': 'FL002', 'task_name': 'boarding', 'duration': 15, 
             'required_vehicles': ['passenger_stairs'], 'predecessors': ['catering']},
        ],
        'statistics': {
            'total_tasks': 8,
            'avg_delay': 0,
            'equipment_failures': 0
        }
    }


def demo_simple_simulation():
    """Demo with simple, known-good scenario"""
    
    print("="*70)
    print("SIMPLE SIMULATION DEMO")
    print("="*70)
    
    # Create simple scenario
    scenario = create_sample_scenario()
    
    print(f"\n✅ Created test scenario")
    print(f"   Flights: {scenario['num_flights']}")
    print(f"   Tasks: {len(scenario['tasks'])}")
    
    # Vehicle configuration
    vehicle_config = {
        'fuel_truck': {
            'count': 2,
            'max_tasks_before_base': 10,
            'compatible_aircraft': ['all']
        },
        'catering_truck': {
            'count': 2,
            'max_tasks_before_base': 4,
            'compatible_aircraft': ['all']
        },
        'passenger_stairs': {
            'count': 2,
            'max_tasks_before_base': 15,
            'compatible_aircraft': ['all']
        }
    }
    
    # Travel times
    positions = ['base', 'gate_1', 'gate_2']
    travel_times = pd.DataFrame(5.0, index=positions, columns=positions)
    for pos in positions:
        travel_times.loc[pos, pos] = 0
    
    # Run simulation
    print("\n🚀 Starting simulation...")
    
    simulation = SimulationEngine(
        scenario=scenario,
        vehicle_config=vehicle_config,
        travel_times=travel_times
    )
    
    results = simulation.run(max_rounds=15)
    
    # Show results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Rounds: {results['rounds']}")
    print(f"Assignments: {results['assignments']}")
    print(f"Conflicts: {results['conflicts']}")
    
    # Show aircraft completion
    print("\n📊 Aircraft Status:")
    for aircraft in simulation.aircraft_agents:
        status = aircraft.get_status()
        print(f"   {aircraft.flight_id}: {status['completion_percent']:.0f}% complete "
              f"({status['completed']}/{status['total_tasks']} tasks)")
    
    # Show vehicle utilization
    print("\n📊 Vehicle Utilization:")
    for vehicle in simulation.vehicle_agents[:6]:  # Show first 6
        status = vehicle.get_status()
        print(f"   {vehicle.vehicle_id}: {status['utilization']:.0f}% "
              f"({status['tasks_completed']}/{status['max_tasks']} tasks)")


def main():
    """Main demo function"""
    
    print("\n")
    print("="*70)
    print("MULTI-AGENT SYSTEM - FIXED DEMO")
    print("="*70)
    
    demo_simple_simulation()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()