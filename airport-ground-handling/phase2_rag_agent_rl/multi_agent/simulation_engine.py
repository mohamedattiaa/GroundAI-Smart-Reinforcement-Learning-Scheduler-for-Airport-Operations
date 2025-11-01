"""
Simulation Engine for Multi-Agent System
"""

import pandas as pd
from typing import List, Dict
from .aircraft_agent import AircraftAgent
from .vehicle_agent import VehicleAgent
from .coordinator_agent import CoordinatorAgent


class SimulationEngine:
    """Run multi-agent simulation"""
    
    def __init__(
        self,
        scenario: Dict,
        vehicle_config: Dict,
        travel_times: pd.DataFrame
    ):
        self.scenario = scenario
        self.vehicle_config = vehicle_config
        self.travel_times = travel_times
        
        # Initialize agents
        self.aircraft_agents = self._create_aircraft_agents()
        self.vehicle_agents = self._create_vehicle_agents()
        self.coordinator = CoordinatorAgent(
            self.aircraft_agents,
            self.vehicle_agents,
            travel_times
        )
        
        # Simulation state
        self.current_time = pd.Timestamp.now()
        self.max_rounds = 100
    
    def _create_aircraft_agents(self) -> List[AircraftAgent]:
        """Create aircraft agents from scenario"""
        
        agents = []
        
        for flight in self.scenario['flights']:
            agent = AircraftAgent(
                flight_id=flight['flight_id'],
                aircraft_type=flight['aircraft_type'],
                position=flight['position'],
                arrival_time=pd.to_datetime(flight['actual_arrival']),
                scheduled_departure=pd.to_datetime(flight['scheduled_departure']),
                tasks=self._get_flight_tasks(flight['flight_id'])
            )
            agents.append(agent)
        
        return agents
    
    def _get_flight_tasks(self, flight_id: str) -> List[Dict]:
        """Get tasks for a specific flight"""
        
        tasks = []
        
        for task in self.scenario['tasks']:
            if task['flight_id'] == flight_id:
                # Ensure task has required fields
                task_dict = {
                    'task_name': task.get('task_name', 'unknown'),
                    'duration': int(task.get('duration', 10)),
                    'required_vehicles': task.get('required_vehicles', []),
                    'predecessors': task.get('predecessors', [])
                }
                
                # Convert string fields to lists if needed
                if isinstance(task_dict['required_vehicles'], str):
                    task_dict['required_vehicles'] = [
                        v.strip() for v in task_dict['required_vehicles'].split(',') if v.strip()
                    ]
                
                if isinstance(task_dict['predecessors'], str):
                    task_dict['predecessors'] = [
                        p.strip() for p in task_dict['predecessors'].split(',') if p.strip()
                    ]
                
                tasks.append(task_dict)
        
        return tasks
    
    def _create_vehicle_agents(self) -> List[VehicleAgent]:
        """Create vehicle agents from config"""
        
        agents = []
        
        for vehicle_type, config in self.vehicle_config.items():
            for i in range(config['count']):
                agent = VehicleAgent(
                    vehicle_id=f"{vehicle_type}_{i+1}",
                    vehicle_type=vehicle_type,
                    max_tasks=config['max_tasks_before_base'],
                    compatible_aircraft=config['compatible_aircraft']
                )
                agents.append(agent)
        
        return agents
    
    def run(self, max_rounds: int = None) -> Dict:
        """Run simulation"""
        
        if max_rounds:
            self.max_rounds = max_rounds
        
        print(f"\n{'='*60}")
        print("STARTING MULTI-AGENT SIMULATION")
        print(f"{'='*60}")
        print(f"Aircraft: {len(self.aircraft_agents)}")
        print(f"Vehicles: {len(self.vehicle_agents)}")
        print(f"Max Rounds: {self.max_rounds}\n")
        
        for round_num in range(self.max_rounds):
            # Mark completed tasks BEFORE scheduling
            self._update_task_completion()
            
            # Run scheduling round
            result = self.coordinator.run_scheduling_round()
            
            if result['requests'] == 0:
                print(f"\n✅ Round {round_num + 1}: No more requests. Simulation complete!")
                break
            
            print(f"Round {round_num + 1}: "
                f"Requests={result['requests']}, "
                f"Assigned={result['assignments']}, "
                f"Conflicts={result['conflicts']}")
            
            # Simulate time passing (instant completion for demo)
            for assignment in self.coordinator.assignments:
                if not assignment.get('completed'):
                    aircraft = self.coordinator.aircraft_agents[assignment['flight_id']]
                    aircraft.complete_task(assignment['task_name'])
                    assignment['completed'] = True
        
        # Final report
        print(f"\n{self.coordinator.generate_report()}")
        
        return {
            'rounds': round_num + 1,
            'assignments': len(self.coordinator.assignments),
            'conflicts': len(self.coordinator.conflicts),
            'status': self.coordinator.get_overall_status()
        }
    
    def _update_task_completion(self):
        """Update task completion status"""
        
        current_time = pd.Timestamp.now()
        
        for assignment in self.coordinator.assignments:
            if assignment.get('completed'):
                continue
            
            # Check if task should be completed
            end_time = assignment['start_time'] + pd.Timedelta(minutes=assignment['duration'])
            
            if current_time >= end_time:
                # Mark as completed
                assignment['completed'] = True
                
                # Update aircraft
                aircraft = self.coordinator.aircraft_agents[assignment['flight_id']]
                aircraft.complete_task(assignment['task_name'])