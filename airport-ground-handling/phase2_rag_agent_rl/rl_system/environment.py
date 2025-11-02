"""
Improved Airport Ground Handling Environment with better reward shaping and dynamics.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Any, Optional


class AirportGroundHandlingEnv(gym.Env):
    """
    Multi-agent airport ground handling optimization environment.
    Tasks: Fueling, Catering, Cleaning
    Goals: Minimize delays and maximize resource utilization
    """
    
    metadata = {'render_modes': []}
    
    def __init__(
        self,
        num_aircraft: int = 10,
        num_vehicles: int = 30,
        episode_length: int = 100,
        reward_config: Optional[Dict] = None,
        task_config: Optional[Dict] = None,
        seed: int = 42
    ):
        """Initialize environment."""
        super().__init__()
        
        self.num_aircraft = num_aircraft
        self.num_vehicles = num_vehicles
        self.max_steps = episode_length
        self.current_step = 0
        
        # Default configs
        self.reward_config = reward_config or self._default_reward_config()
        self.task_config = task_config or self._default_task_config()
        
        # Random seed
        np.random.seed(seed)
        
        # Define observation and action spaces
        self._setup_spaces()
        
        # Initialize state
        self.reset()
    
    def _default_reward_config(self) -> Dict:
        """Default reward configuration."""
        return {
            'task_completion': 50.0,
            'delay_penalty_per_step': -1.0,
            'idle_penalty': -0.5,
            'action_efficiency': 10.0,
            'collision_penalty': -100.0,
        }
    
    def _default_task_config(self) -> Dict:
        """Default task configuration."""
        return {
            'fueling': {
                'duration_min': 5, 'duration_max': 15,
                'priority': 1.0, 'resource_requirement': 1
            },
            'catering': {
                'duration_min': 10, 'duration_max': 25,
                'priority': 0.8, 'resource_requirement': 1
            },
            'cleaning': {
                'duration_min': 15, 'duration_max': 30,
                'priority': 0.6, 'resource_requirement': 1
            }
        }
    
    def _setup_spaces(self):
        """Setup observation and action spaces."""
        # Observation space: Dict with aircraft, global, and vehicle info
        self.observation_space = spaces.Dict({
            'aircraft': spaces.Box(0, 1, shape=(self.num_aircraft, 8), dtype=np.float32),
            'global': spaces.Box(0, 1, shape=(5,), dtype=np.float32),
            'vehicles': spaces.Box(0, 1, shape=(self.num_vehicles, 6), dtype=np.float32)
        })
        
        # Action space: discrete actions for each aircraft and vehicle
        # [select_aircraft, select_task, select_vehicle]
        self.action_space = spaces.MultiDiscrete([
            self.num_aircraft,      # Which aircraft to service
            len(self.task_config),  # Which task (fueling/catering/cleaning)
            self.num_vehicles       # Which vehicle to assign
        ])
        
        self.task_names = list(self.task_config.keys())
    
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict, Dict]:
        """Reset environment."""
        if seed is not None:
            np.random.seed(seed)
        
        self.current_step = 0
        
        # Initialize aircraft states: [id, arrival_time, tasks_needed, priority, urgency, 
        #                              fueling_done, catering_done, cleaning_done]
        self.aircraft_states = np.random.uniform(0, 1, (self.num_aircraft, 8)).astype(np.float32)
        
        # Initialize global state: [total_delay, avg_utilization, available_resources, 
        #                          episode_progress, task_completion_rate]
        self.global_state = np.array([0, 0.5, 1.0, 0, 0], dtype=np.float32)
        
        # Initialize vehicle states: [id, availability, task_type, progress, 
        #                             efficiency, idle_time]
        self.vehicle_states = np.random.uniform(0, 1, (self.num_vehicles, 6)).astype(np.float32)
        
        # Tracking metrics
        self.total_delay = 0.0
        self.tasks_completed = 0
        self.episode_info = {
            'total_reward': 0,
            'total_delay': 0,
            'episode_length': 0,
            'tasks_completed': 0,
            'resource_utilization': 0,
            'collisions': 0
        }
        
        return self._get_observation(), {}
    
    def step(self, action: np.ndarray) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one environment step."""
        self.current_step += 1
        
        # Parse action
        aircraft_id = action[0]
        task_id = action[1]
        vehicle_id = action[2]
        
        # Calculate reward components
        reward = 0.0
        info = {}
        
        # 1. Check for valid assignment
        if self._is_valid_assignment(aircraft_id, task_id, vehicle_id):
            reward += self.reward_config['action_efficiency']
            info['valid_action'] = True
            
            # 2. Check for collision/conflict
            if self._check_collision(aircraft_id, vehicle_id):
                reward += self.reward_config['collision_penalty']
                info['collision'] = True
                self.episode_info['collisions'] += 1
            
            # 3. Execute task assignment
            task_duration = self._assign_task(aircraft_id, task_id, vehicle_id)
            
            # 4. Task completion reward
            if task_duration > 0:
                reward += self.reward_config['task_completion']
                self.tasks_completed += 1
                self.episode_info['tasks_completed'] += 1
                info['task_completed'] = True
            
            # 5. Efficiency bonus based on urgency
            urgency = self.aircraft_states[aircraft_id, 4]
            reward += urgency * 5.0  # Bonus for urgent tasks
        
        else:
            reward += self.reward_config['action_efficiency'] * 0.1  # Small penalty for invalid
            info['valid_action'] = False
        
        # 6. Delay penalty for idle resources
        idle_vehicles = np.sum(self.vehicle_states[:, 1] > 0.9)  # Vehicles with >90% availability
        reward += self.reward_config['idle_penalty'] * idle_vehicles
        
        # 7. Update delays
        self.total_delay += self._calculate_current_delays()
        reward += self.reward_config['delay_penalty_per_step'] * (self.total_delay / (self.current_step + 1))
        
        # Update states
        self._update_states()
        self.global_state[0] = self.total_delay / max(1, self.current_step)
        self.global_state[3] = self.current_step / self.max_steps
        
        # Track episode info
        self.episode_info['total_reward'] += reward
        self.episode_info['total_delay'] = self.total_delay
        self.episode_info['episode_length'] = self.current_step
        self.episode_info['resource_utilization'] = 1.0 - (idle_vehicles / self.num_vehicles)
        
        info['delay'] = self.total_delay
        info['tasks_completed'] = self.tasks_completed
        
        # Check termination
        terminated = self.current_step >= self.max_steps
        
        return self._get_observation(), reward, terminated, False, info
    
    def _is_valid_assignment(self, aircraft_id: int, task_id: int, vehicle_id: int) -> bool:
        """Check if assignment is valid."""
        if not (0 <= aircraft_id < self.num_aircraft):
            return False
        if not (0 <= task_id < len(self.task_names)):
            return False
        if not (0 <= vehicle_id < self.num_vehicles):
            return False
        
        # Check if vehicle is available
        if self.vehicle_states[vehicle_id, 1] < 0.3:  # Less than 30% available
            return False
        
        return True
    
    def _check_collision(self, aircraft_id: int, vehicle_id: int) -> bool:
        """Check if assignment would cause resource conflict."""
        # Check if vehicle is already assigned (simplified)
        return self.vehicle_states[vehicle_id, 1] < 0.1
    
    def _assign_task(self, aircraft_id: int, task_id: int, vehicle_id: int) -> float:
        """Assign a task and return duration."""
        task_name = self.task_names[task_id]
        task_cfg = self.task_config[task_name]
        
        # Determine if task is already done
        task_field = 5 + task_id  # Fields 5, 6, 7 for fueling, catering, cleaning
        if self.aircraft_states[aircraft_id, task_field] > 0.9:
            return 0  # Task already completed
        
        # Simulate task duration
        duration = np.random.uniform(
            task_cfg['duration_min'] / 100,
            task_cfg['duration_max'] / 100
        )  # Normalized
        
        # Update states
        self.aircraft_states[aircraft_id, task_field] += duration
        self.vehicle_states[vehicle_id, 1] -= 0.1  # Decrease availability
        self.vehicle_states[vehicle_id, 2] = task_id  # Current task
        self.vehicle_states[vehicle_id, 3] += duration  # Progress
        
        return duration
    
    def _calculate_current_delays(self) -> float:
        """Calculate delay based on incomplete tasks."""
        incomplete_tasks = 0
        for i in range(self.num_aircraft):
            # Check if any task is incomplete
            for j in range(3):  # 3 task types
                if self.aircraft_states[i, 5 + j] < 0.9:
                    incomplete_tasks += 1
        
        return incomplete_tasks * 0.1
    
    def _update_states(self):
        """Update environment states for next step."""
        # Decay vehicle unavailability (recovery)
        self.vehicle_states[:, 1] = np.minimum(
            self.vehicle_states[:, 1] + 0.02,
            1.0
        )
        
        # Increase idle time for vehicles not in use
        idle_mask = self.vehicle_states[:, 1] > 0.8
        self.vehicle_states[idle_mask, 5] += 0.1
        self.vehicle_states[~idle_mask, 5] = 0
        
        # Update global utilization
        avg_util = 1.0 - np.mean(self.vehicle_states[:, 1])
        self.global_state[1] = avg_util
    
    def _get_observation(self) -> Dict:
        """Get current observation."""
        return {
            'aircraft': self.aircraft_states,
            'global': self.global_state,
            'vehicles': self.vehicle_states
        }
    
    def get_episode_done(self) -> bool:
        """Check if episode is done (for callback)."""
        return self.current_step >= self.max_steps
    
    def get_episode_info(self) -> Dict:
        """Get episode information (for callback)."""
        return self.episode_info
    
    def render(self):
        """Render environment state."""
        print(f"Step: {self.current_step}/{self.max_steps}")
        print(f"Total Delay: {self.total_delay:.2f}")
        print(f"Tasks Completed: {self.tasks_completed}")
        print(f"Avg Resource Util: {self.global_state[1]:.2f}")