#!/usr/bin/env python3
"""
Test script for Multi-Agent System
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from phase2_rag_agent_rl.multi_agent.aircraft_agent import AircraftAgent
from phase2_rag_agent_rl.multi_agent.vehicle_agent import VehicleAgent
from phase2_rag_agent_rl.multi_agent.coordinator_agent import CoordinatorAgent


def test_aircraft_agent():
    """Test aircraft agent functionality"""
    
    print("\n🧪 Testing Aircraft Agent...")
    
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
        }
    ]
    
    aircraft = AircraftAgent(
        flight_id='TEST001',
        aircraft_type='A320',
        position='gate_1',
        arrival_time=pd.Timestamp.now(),
        scheduled_departure=pd.Timestamp.now() + pd.Timedelta(minutes=45),
        tasks=tasks
    )
    
    # Test 1: Get next tasks
    ready = aircraft.get_next_tasks()
    assert len(ready) == 1, "Should have 1 ready task"
    assert ready[0]['task_name'] == 'deplaning', "First task should be deplaning"
    
    # Test 2: Complete task
    aircraft.complete_task('deplaning')
    ready = aircraft.get_next_tasks()
    assert len(ready) == 1, "Should have 1 ready task after completing deplaning"
    assert ready[0]['task_name'] == 'refueling', "Next task should be refueling"
    
    # Test 3: Status
    status = aircraft.get_status()
    assert status['completed'] == 1, "Should have 1 completed task"
    assert status['remaining'] == 1, "Should have 1 remaining task"
    
    print("✅ Aircraft Agent tests passed")
    return True


def test_vehicle_agent():
    """Test vehicle agent functionality"""
    
    print("\n🧪 Testing Vehicle Agent...")
    
    vehicle = VehicleAgent(
        vehicle_id='TEST_VEHICLE',
        vehicle_type='fuel_truck',
        max_tasks=10,
        compatible_aircraft=['all']
    )
    
    # Test 1: Can serve check
    request = {
        'flight_id': 'FL001',
        'aircraft_type': 'A320',
        'position': 'gate_5',
        'task_name': 'refueling',
        'duration': 12,
        'required_vehicles': ['fuel_truck']
    }
    
    assert vehicle.can_serve(request), "Should be able to serve refueling request"
    
    # Test 2: Bidding
    bid = vehicle.bid_on_request(request, travel_time=5)
    assert bid is not None, "Should generate a bid"
    assert 'bid_score' in bid, "Bid should have score"
    
    # Test 3: Task assignment
    vehicle.assign_task(request, pd.Timestamp.now(), 12)
    assert vehicle.tasks_completed == 1, "Should have 1 completed task"
    assert len(vehicle.schedule) == 1, "Schedule should have 1 task"
    
    print("✅ Vehicle Agent tests passed")
    return True


def test_coordinator():
    """Test coordinator agent"""
    
    print("\n🧪 Testing Coordinator Agent...")
    
    # Create test agents
    aircraft = AircraftAgent(
        flight_id='FL001',
        aircraft_type='A320',
        position='gate_1',
        arrival_time=pd.Timestamp.now(),
        scheduled_departure=pd.Timestamp.now() + pd.Timedelta(minutes=45),
        tasks=[{
            'task_name': 'refueling',
            'duration': 12,
            'required_vehicles': ['fuel_truck'],
            'predecessors': []
        }]
    )
    
    vehicle = VehicleAgent(
        vehicle_id='fuel_truck_1',
        vehicle_type='fuel_truck',
        max_tasks=10,
        compatible_aircraft=['all']
    )
    
    # Create travel times
    travel_times = pd.DataFrame(
        [[0, 5], [5, 0]],
        index=['base', 'gate_1'],
        columns=['base', 'gate_1']
    )
    
    coordinator = CoordinatorAgent(
        aircraft_agents=[aircraft],
        vehicle_agents=[vehicle],
        travel_times=travel_times
    )
    
    # Test scheduling round
    result = coordinator.run_scheduling_round()
    
    assert result['requests'] >= 0, "Should have requests count"
    assert result['assignments'] >= 0, "Should have assignments count"
    
    print("✅ Coordinator tests passed")
    return True


def run_all_tests():
    """Run all tests"""
    
    print("="*60)
    print("MULTI-AGENT SYSTEM TESTS")
    print("="*60)
    
    tests = [
        test_aircraft_agent,
        test_vehicle_agent,
        test_coordinator
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"❌ Test failed: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ Error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)