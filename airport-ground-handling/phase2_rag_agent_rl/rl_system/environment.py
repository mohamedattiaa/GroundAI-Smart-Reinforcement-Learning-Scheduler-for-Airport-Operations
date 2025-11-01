"""
Gymnasium Environment for Airport Ground Handling
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class AirportEnv(gym.Env):
    """
    Custom Gymnasium Environment for Airport Ground Handling
    
    The agent learns to assign vehicles to aircraft tasks to minimize delay
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        scenario: Dict,
        vehicle_config: Dict,
        max_steps: int = 200
    ):
        super().__init__()
        
        self.scenario = scenario
        self.vehicle_config = vehicle_config
        self.max_steps = max_steps
        
        # Extract data
        self.flights = pd.DataFrame(scenario['flights'])
        self.tasks_data = pd.DataFrame(scenario['tasks'])
        
        # State dimensions
        self.max_aircraft = 10
        self.max_vehicles = 30
        self.max_tasks_per_aircraft = 10
        
        # Define observation space
        # State = [aircraft_states, vehicle_states, global_state]
        self.observation_space = spaces.Dict({
            'aircraft': spaces.Box(
                low=0, high=1,
                shape=(self.max_aircraft, 8),
                dtype=np.float32
            ),
            'vehicles': spaces.Box(
                low=0, high=1,
                shape=(self.max_vehicles, 6),
                dtype=np.float32
            ),
            'global': spaces.Box(
                low=0, high=1,
                shape=(5,),
                dtype=np.float32
            )
        })
        
        # Define action space
        # Action = [aircraft_id, task_id, vehicle_id]
        self.action_space = spaces.MultiDiscrete([
            self.max_aircraft,  # Which aircraft
            self.max_tasks_per_aircraft,  # Which task
            self.max_vehicles  # Which vehicle
        ])
        
        # Initialize state
        self.reset()
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[Dict, Dict]:
        """Reset environment to initial state"""
        
        super().reset(seed=seed)
        
        # Reset tracking
        self.current_step = 0
        self.total_delay = 0.0
        self.assignments_made = 0
        
        # Initialize aircraft states
        self.aircraft_states = []
        for i, flight in enumerate(self.flights.itertuples()):
            if i >= self.max_aircraft:
                break
            
            aircraft_tasks = self.tasks_data[
                self.tasks_data['flight_id'] == flight.flight_id
            ]
            
            self.aircraft_states.append({
                'flight_id': flight.flight_id,
                'aircraft_type': flight.aircraft_type,
                'position': flight.position,
                'total_tasks': len(aircraft_tasks),
                'completed_tasks': 0,
                'current_delay': 0.0
            })
        
        # Initialize vehicle states
        self.vehicle_states = []
        vehicle_id = 0
        for v_type, config in self.vehicle_config.items():
            for i in range(config['count']):
                if vehicle_id >= self.max_vehicles:
                    break
                
                self.vehicle_states.append({
                    'vehicle_id': f"{v_type}_{i}",
                    'vehicle_type': v_type,
                    'position': 'base',
                    'available_at': 0.0,
                    'tasks_completed': 0,
                    'max_tasks': config['max_tasks_before_base']
                })
                vehicle_id += 1
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        
        aircraft_idx, task_idx, vehicle_idx = action
        
        # Validate action
        if not self._is_valid_action(aircraft_idx, task_idx, vehicle_idx):
            reward = -10.0  # Penalty for invalid action
            observation = self._get_observation()
            terminated = False
            truncated = self.current_step >= self.max_steps
            info = self._get_info()
            return observation, reward, terminated, truncated, info
        
        # Execute assignment
        reward = self._execute_assignment(aircraft_idx, task_idx, vehicle_idx)
        
        # Update state
        self.current_step += 1
        self.assignments_made += 1
        
        # Check if done
        terminated = self._check_terminated()
        truncated = self.current_step >= self.max_steps
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _is_valid_action(
        self,
        aircraft_idx: int,
        task_idx: int,
        vehicle_idx: int
    ) -> bool:
        """Check if action is valid"""
        
        # Check indices are in range
        if aircraft_idx >= len(self.aircraft_states):
            return False
        if vehicle_idx >= len(self.vehicle_states):
            return False
        
        aircraft = self.aircraft_states[aircraft_idx]
        vehicle = self.vehicle_states[vehicle_idx]
        
        # Check aircraft has remaining tasks
        if aircraft['completed_tasks'] >= aircraft['total_tasks']:
            return False
        
        # Check vehicle has capacity
        if vehicle['tasks_completed'] >= vehicle['max_tasks']:
            return False
        
        return True
    
    def _execute_assignment(
        self,
        aircraft_idx: int,
        task_idx: int,
        vehicle_idx: int
    ) -> float:
        """Execute vehicle assignment and return reward"""
        
        aircraft = self.aircraft_states[aircraft_idx]
        vehicle = self.vehicle_states[vehicle_idx]
        
        # Simulate task execution
        task_duration = 10  # Simplified
        travel_time = 5  # Simplified
        
        # Calculate delay
        start_time = vehicle['available_at'] + travel_time
        expected_start = self.current_step * 2  # Expected timeline
        delay = max(0, start_time - expected_start)
        
        # Update states
        aircraft['completed_tasks'] += 1
        aircraft['current_delay'] += delay
        
        vehicle['available_at'] = start_time + task_duration
        vehicle['tasks_completed'] += 1
        vehicle['position'] = aircraft['position']
        
        # Calculate reward
        # Positive reward for completion, negative for delay
        completion_reward = 10.0
        delay_penalty = -delay * 0.5
        
        # Bonus for completing all aircraft tasks
        if aircraft['completed_tasks'] == aircraft['total_tasks']:
            completion_reward += 20.0
        
        reward = completion_reward + delay_penalty
        
        self.total_delay += delay
        
        return reward
    
    def _check_terminated(self) -> bool:
        """Check if episode is complete"""
        
        # Episode ends when all aircraft have completed all tasks
        all_complete = all(
            ac['completed_tasks'] >= ac['total_tasks']
            for ac in self.aircraft_states
        )
        
        return all_complete
    
    def _get_observation(self) -> Dict:
        """Get current observation"""
        
        # Aircraft observations (normalized)
        aircraft_obs = np.zeros((self.max_aircraft, 8), dtype=np.float32)
        for i, aircraft in enumerate(self.aircraft_states):
            if i < self.max_aircraft:
                aircraft_obs[i] = [
                    aircraft['completed_tasks'] / max(aircraft['total_tasks'], 1),
                    aircraft['current_delay'] / 100.0,  # Normalize delay
                    float(aircraft['aircraft_type'] == 'A320'),
                    float(aircraft['aircraft_type'] == 'B737'),
                    float(aircraft['aircraft_type'] == 'B777'),
                    float(aircraft['aircraft_type'] == 'A321'),
                    float(aircraft['aircraft_type'] == 'A350'),
                    self.current_step / self.max_steps
                ]
        
        # Vehicle observations (normalized)
        vehicle_obs = np.zeros((self.max_vehicles, 6), dtype=np.float32)
        for i, vehicle in enumerate(self.vehicle_states):
            if i < self.max_vehicles:
                vehicle_obs[i] = [
                    vehicle['available_at'] / 200.0,  # Normalize time
                    vehicle['tasks_completed'] / max(vehicle['max_tasks'], 1),
                    float(vehicle['vehicle_type'] == 'fuel_truck'),
                    float(vehicle['vehicle_type'] == 'baggage_loader'),
                    float(vehicle['vehicle_type'] == 'catering_truck'),
                    float(vehicle['position'] == 'base')
                ]
        
        # Global observations
        global_obs = np.array([
            self.current_step / self.max_steps,
            self.assignments_made / (self.max_aircraft * 10),
            self.total_delay / 1000.0,
            len([ac for ac in self.aircraft_states 
                 if ac['completed_tasks'] >= ac['total_tasks']]) / max(len(self.aircraft_states), 1),
            len([v for v in self.vehicle_states 
                 if v['tasks_completed'] < v['max_tasks']]) / max(len(self.vehicle_states), 1)
        ], dtype=np.float32)
        
        return {
            'aircraft': aircraft_obs,
            'vehicles': vehicle_obs,
            'global': global_obs
        }
    
    def _get_info(self) -> Dict:
        """Get additional info"""
        
        return {
            'step': self.current_step,
            'assignments': self.assignments_made,
            'total_delay': self.total_delay,
            'completed_aircraft': sum(
                1 for ac in self.aircraft_states
                if ac['completed_tasks'] >= ac['total_tasks']
            )
        }
    
    def render(self, mode='human'):
        """Render the environment"""
        
        if mode == 'human' or mode == 'ansi':
            output = []
            output.append(f"\nStep: {self.current_step}/{self.max_steps}")
            output.append(f"Assignments: {self.assignments_made}")
            output.append(f"Total Delay: {self.total_delay:.2f}")
            output.append(f"\nAircraft:")
            
            for i, ac in enumerate(self.aircraft_states[:5]):
                output.append(
                    f"  {i+1}. {ac['flight_id']}: "
                    f"{ac['completed_tasks']}/{ac['total_tasks']} tasks, "
                    f"delay: {ac['current_delay']:.1f}"
                )
            
            return '\n'.join(output)