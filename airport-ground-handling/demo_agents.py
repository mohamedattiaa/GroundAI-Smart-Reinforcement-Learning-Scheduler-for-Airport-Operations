#!/usr/bin/env python3
"""
Demo script for Multi-Agent System

This demonstrates:
1. Creating aircraft and vehicle agents
2. Coordinating resource allocation
3. Running multi-agent simulation
4. Analyzing results
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


def demo_basic_agents():
    """Demo: Basic agent creation and interaction"""
    
    print("="*70)
    print("DEMO 1: Basic Agents")
    print("="*70)
    
    # Create an aircraft agent
    print("\n1️⃣  Creating Aircraft Agent...")
    
    tasks = [
        {
            'task_name': 'deplaning',
            'duration': 7,
            'required_vehicles': ['passenger_stairs'],
            'predecessors': []
        },
        {
            'task_name': 'refueling',
            'duration': 12,
            'required_vehicles': ['fuel_truck'],
            'predecessors': ['deplaning']
        },
        {
            'task_name': 'boarding',
            'duration': 15,
            'required_vehicles': ['passenger_stairs'],
            'predecessors': ['deplaning']
        }
    ]
    
    aircraft = AircraftAgent(
        flight_id='FL0001',
        aircraft_type='A320',
        position='gate_1',
        arrival_time=pd.Timestamp.now(),
        scheduled_departure=pd.Timestamp.now() + pd.Timedelta(minutes=45),
        tasks=tasks
    )
    
    print(aircraft.to_message())
    
    # Create vehicle agents
    print("\n2️⃣  Creating Vehicle Agents...")
    
    fuel_truck = VehicleAgent(
        vehicle_id='fuel_truck_1',
        vehicle_type='fuel_truck',
        max_tasks=10,
        compatible_aircraft=['all']
    )
    
    stairs = VehicleAgent(
        vehicle_id='stairs_1',
        vehicle_type='passenger_stairs',
        max_tasks=15,
        compatible_aircraft=['A320', 'B737']
    )
    
    print(fuel_truck.to_message())
    print(stairs.to_message())
    
    # Request service
    print("\n3️⃣  Aircraft Requesting Services...")
    
    ready_tasks = aircraft.get_next_tasks()
    print(f"\nReady tasks: {[t['task_name'] for t in ready_tasks]}")
    
    for task in ready_tasks:
        request = aircraft.request_service(task)
        print(f"\n📋 Service Request:")
        print(f"   Task: {request['task_name']}")
        print(f"   Duration: {request['duration']} min")
        print(f"   Priority: {request['priority']:.2f}")
        print(f"   Urgency: {request['urgency']:.2f}")
    
    input("\n\nPress Enter to continue...")


def demo_bidding_system():
    """Demo: Vehicle bidding on requests"""
    
    print("\n" + "="*70)
    print("DEMO 2: Bidding System")
    print("="*70)
    
    # Create request
    request = {
        'flight_id': 'FL0001',
        'aircraft_type': 'A320',
        'position': 'gate_5',
        'task_name': 'refueling',
        'duration': 12,
        'required_vehicles': ['fuel_truck'],
        'priority': 1.2,
        'urgency': 0.7
    }
    
    print(f"\n📋 Service Request: {request['task_name']} for {request['flight_id']}")
    
    # Create multiple vehicles
    vehicles = []
    for i in range(3):
        v = VehicleAgent(
            vehicle_id=f'fuel_truck_{i+1}',
            vehicle_type='fuel_truck',
            max_tasks=10,
            compatible_aircraft=['all']
        )
        # Set different states
        v.current_position = ['base', 'gate_3', 'gate_8'][i]
        v.tasks_completed = [0, 3, 7][i]
        vehicles.append(v)
    
    # Collect bids
    print("\n💰 Collecting Bids...")
    
    bids = []
    for vehicle in vehicles:
        # Simulate travel time
        travel_times = {'base': 5, 'gate_3': 3, 'gate_8': 7}
        travel_time = travel_times.get(vehicle.current_position, 5)
        
        bid = vehicle.bid_on_request(request, travel_time)
        
        if bid:
            bids.append(bid)
            print(f"\n✅ {bid['vehicle_id']}:")
            print(f"   Position: {bid['current_position']}")
            print(f"   Available: {bid['available_at']}")
            print(f"   Bid Score: {bid['bid_score']:.2f}")
            print(f"   Workload: {bid['tasks_completed']}/10")
    
    # Select best bid
    if bids:
        bids.sort(key=lambda b: b['bid_score'])
        winner = bids[0]
        print(f"\n🏆 Winner: {winner['vehicle_id']} (Score: {winner['bid_score']:.2f})")
    
    input("\n\nPress Enter to continue...")


def demo_coordination():
    """Demo: Coordinator managing multiple aircraft and vehicles"""
    
    print("\n" + "="*70)
    print("DEMO 3: Coordinator Agent")
    print("="*70)
    
    # Load real scenario
    print("\n📂 Loading scenario...")
    loader = DatasetLoader(data_dir="data/raw")
    scenarios = loader.load_scenario_files()
    
    if not scenarios:
        print("❌ No scenarios found. Please generate data first.")
        return
    
    scenario = scenarios[0]  # Use first scenario
    
    print(f"✅ Loaded scenario: {scenario['scenario_id']}")
    print(f"   Flights: {scenario['num_flights']}")
    print(f"   Tasks: {scenario['statistics']['total_tasks']}")
    
    # Create aircraft agents (limit to first 3 for demo)
    print("\n1️⃣  Creating Aircraft Agents...")
    
    aircraft_agents = []
    for flight in scenario['flights'][:3]:
        # Get tasks for this flight
        flight_tasks = [
            task for task in scenario['tasks']
            if task['flight_id'] == flight['flight_id']
        ]
        
        agent = AircraftAgent(
            flight_id=flight['flight_id'],
            aircraft_type=flight['aircraft_type'],
            position=flight['position'],
            arrival_time=pd.to_datetime(flight['actual_arrival']),
            scheduled_departure=pd.to_datetime(flight['scheduled_departure']),
            tasks=flight_tasks
        )
        aircraft_agents.append(agent)
        print(f"   ✅ {agent.flight_id} ({agent.aircraft_type})")
    
    # Create vehicle agents
    print("\n2️⃣  Creating Vehicle Agents...")
    
    vehicle_config = {
        'fuel_truck': {'count': 2, 'max_tasks': 10, 'compatible': ['all']},
        'baggage_loader': {'count': 3, 'max_tasks': 12, 'compatible': ['all']},
        'passenger_stairs': {'count': 2, 'max_tasks': 15, 'compatible': ['A320', 'B737']}
    }
    
    vehicle_agents = []
    for v_type, config in vehicle_config.items():
        for i in range(config['count']):
            agent = VehicleAgent(
                vehicle_id=f"{v_type}_{i+1}",
                vehicle_type=v_type,
                max_tasks=config['max_tasks'],
                compatible_aircraft=config['compatible']
            )
            vehicle_agents.append(agent)
        print(f"   ✅ {config['count']} {v_type}")
    
    # Create travel times (simplified)
    print("\n3️⃣  Setting up travel times...")
    
    positions = list(set([a.position for a in aircraft_agents] + ['base']))
    travel_times = pd.DataFrame(5.0, index=positions, columns=positions)
    for pos in positions:
        travel_times.loc[pos, pos] = 0
    
    # Create coordinator
    print("\n4️⃣  Creating Coordinator...")
    
    coordinator = CoordinatorAgent(
        aircraft_agents,
        vehicle_agents,
        travel_times
    )
    
    print("✅ Coordinator initialized")
    
    # Run scheduling rounds
    print("\n5️⃣  Running Scheduling Rounds...")
    
    for round_num in range(5):
        result = coordinator.run_scheduling_round()
        
        if result['requests'] == 0:
            print(f"\n✅ Round {round_num + 1}: No more requests")
            break
        
        print(f"\nRound {round_num + 1}:")
        print(f"   Requests: {result['requests']}")
        print(f"   Assigned: {result['assignments']}")
        print(f"   Conflicts: {result['conflicts']}")
    
    # Show final status
    print("\n" + "="*70)
    print("FINAL STATUS")
    print("="*70)
    print(coordinator.generate_report())
    
    input("\nPress Enter to continue...")


def demo_full_simulation():
    """Demo: Complete simulation with real data"""
    
    print("\n" + "="*70)
    print("DEMO 4: Full Simulation")
    print("="*70)
    
    # Load scenario
    print("\n📂 Loading scenario...")
    loader = DatasetLoader(data_dir="data/raw")
    scenarios = loader.load_scenario_files()
    
    if not scenarios:
        print("❌ No scenarios found. Please generate data first.")
        return
    
    # Use first scenario (limit flights for demo)
    scenario = scenarios[0]
    scenario['flights'] = scenario['flights'][:5]  # Only 5 flights for demo
    scenario['tasks'] = [
        t for t in scenario['tasks']
        if t['flight_id'] in [f['flight_id'] for f in scenario['flights']]
    ]
    
    print(f"✅ Scenario: {scenario['scenario_id']}")
    print(f"   Flights: {len(scenario['flights'])}")
    print(f"   Tasks: {len(scenario['tasks'])}")
    
    # Vehicle configuration
    vehicle_config = {
        'fuel_truck': {
            'count': 3,
            'max_tasks_before_base': 10,
            'compatible_aircraft': ['all']
        },
        'baggage_loader': {
            'count': 4,
            'max_tasks_before_base': 12,
            'compatible_aircraft': ['all']
        },
        'catering_truck': {
            'count': 3,
            'max_tasks_before_base': 4,
            'compatible_aircraft': ['all']
        },
        'passenger_stairs': {
            'count': 3,
            'max_tasks_before_base': 15,
            'compatible_aircraft': ['all']
        },
        'cleaning_crew': {
            'count': 2,
            'max_tasks_before_base': 8,
            'compatible_aircraft': ['all']
        }
    }
    
    # Travel times (simplified)
    all_positions = list(set([f['position'] for f in scenario['flights']] + ['base']))
    travel_times = pd.DataFrame(5.0, index=all_positions, columns=all_positions)
    for pos in all_positions:
        travel_times.loc[pos, pos] = 0
    
    # Create and run simulation
    print("\n🚀 Starting simulation...")
    
    simulation = SimulationEngine(
        scenario=scenario,
        vehicle_config=vehicle_config,
        travel_times=travel_times
    )
    
    results = simulation.run(max_rounds=20)
    
    # Analyze results
    print("\n" + "="*70)
    print("SIMULATION RESULTS")
    print("="*70)
    print(f"Rounds completed: {results['rounds']}")
    print(f"Total assignments: {results['assignments']}")
    print(f"Total conflicts: {results['conflicts']}")
    
    if results['conflicts'] > 0:
        print(f"\n⚠️  {results['conflicts']} conflicts occurred")
        print("   Consider: More vehicles or better scheduling")
    else:
        print("\n✅ All requests assigned successfully!")
    
    # Vehicle utilization
    print("\n" + "="*70)
    print("VEHICLE UTILIZATION")
    print("="*70)
    
    for vehicle in simulation.vehicle_agents:
        status = vehicle.get_status()
        print(f"{vehicle.vehicle_id}: {status['utilization']:.1f}% "
              f"({status['tasks_completed']}/{status['max_tasks']} tasks)")


def main():
    """Main demo function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent System Demo")
    parser.add_argument(
        '--mode',
        choices=['basic', 'bidding', 'coordination', 'simulation', 'all'],
        default='all',
        help='Demo mode to run'
    )
    
    args = parser.parse_args()
    
    print("\n")
    print("="*70)
    print("MULTI-AGENT SYSTEM DEMO")
    print("="*70)
    
    if args.mode in ['basic', 'all']:
        demo_basic_agents()
    
    if args.mode in ['bidding', 'all']:
        demo_bidding_system()
    
    if args.mode in ['coordination', 'all']:
        demo_coordination()
    
    if args.mode in ['simulation', 'all']:
        demo_full_simulation()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE")
    print("="*70)
    print("\nTo run specific demos:")
    print("  python demo_agents.py --mode basic")
    print("  python demo_agents.py --mode bidding")
    print("  python demo_agents.py --mode coordination")
    print("  python demo_agents.py --mode simulation")


if __name__ == "__main__":
    main()