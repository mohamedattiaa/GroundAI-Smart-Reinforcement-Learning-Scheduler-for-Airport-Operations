"""
Aircraft schedule and task generation
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class AircraftTask:
    """Represents a single ground handling task"""
    task_name: str
    duration: int  # minutes
    required_vehicles: List[str]
    predecessors: List[str]


class AircraftType:
    """Represents an aircraft type with its tasks and specifications"""
    
    def __init__(self, type_code: str, specs: dict):
        self.type_code = type_code
        self.category = specs['category']
        self.pax_capacity = specs['pax_capacity']
        self.turnaround_time = specs['typical_turnaround']
        
        # Parse tasks
        self.tasks = self._parse_tasks(specs['tasks'])
        self.precedence = specs['precedence']
    
    def _parse_tasks(self, tasks_config: dict) -> Dict[str, AircraftTask]:
        """Parse task configurations"""
        tasks = {}
        
        for task_name, task_data in tasks_config.items():
            tasks[task_name] = AircraftTask(
                task_name=task_name,
                duration=task_data['duration'],
                required_vehicles=task_data['required_vehicles'],
                predecessors=[]  # Will be filled from precedence
            )
        
        return tasks
    
    def get_task_duration_with_variance(self, task_name: str, variance_factor: float = 0.2) -> int:
        """Get task duration with random variance"""
        base_duration = self.tasks[task_name].duration
        variance = base_duration * variance_factor
        actual_duration = np.random.uniform(
            base_duration - variance,
            base_duration + variance
        )
        return int(np.round(actual_duration))


class AircraftGenerator:
    """
    Generates realistic flight schedules and aircraft assignments
    """
    
    def __init__(
        self,
        aircraft_config_path: str = "configs/aircraft_types.yaml",
        generation_config_path: str = "configs/generation_config.yaml"
    ):
        self.aircraft_types = self._load_aircraft_types(aircraft_config_path)
        self.gen_config = self._load_generation_config(generation_config_path)
        
        # Aircraft distribution probabilities
        self.aircraft_probs = self.gen_config['generation']['aircraft_distribution']
    
    def _load_aircraft_types(self, path: str) -> Dict[str, AircraftType]:
        """Load aircraft type specifications"""
        with open(path, 'r') as f:
            config = yaml.safe_load(f)
        
        aircraft_types = {}
        for type_code, specs in config['aircraft_types'].items():
            aircraft_types[type_code] = AircraftType(type_code, specs)
        
        return aircraft_types
    
    def _load_generation_config(self, path: str) -> dict:
        """Load generation parameters"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def generate_daily_schedule(
        self,
        date: datetime,
        airport_config,
        num_flights: int = None
    ) -> pd.DataFrame:
        """
        Generate flight schedule for a single day
        
        Args:
            date: Date to generate schedule for
            airport_config: AirportConfig object
            num_flights: Number of flights (if None, determined by config)
        
        Returns:
            DataFrame with flight schedule
        """
        is_weekend = date.weekday() >= 5
        
        if num_flights is None:
            if is_weekend:
                num_flights = np.random.randint(
                    self.gen_config['generation']['flights_per_day']['weekend']['min'],
                    self.gen_config['generation']['flights_per_day']['weekend']['max']
                )
            else:
                num_flights = np.random.randint(
                    self.gen_config['generation']['flights_per_day']['weekday']['min'],
                    self.gen_config['generation']['flights_per_day']['weekday']['max']
                )
        
        flights = []
        
        for i in range(num_flights):
            # Select aircraft type
            aircraft_type = np.random.choice(
                list(self.aircraft_probs.keys()),
                p=list(self.aircraft_probs.values())
            )
            
            # Generate arrival time (weighted by peak hours)
            arrival_time = self._generate_arrival_time(date)
            
            # Assign position
            compatible_positions = airport_config.get_compatible_positions(aircraft_type)
            position = np.random.choice(compatible_positions)
            
            # Calculate scheduled departure with buffer
            turnaround = self.aircraft_types[aircraft_type].turnaround_time
            # Add 20% buffer to account for potential delays
            scheduled_departure = arrival_time + timedelta(minutes=int(turnaround * 1.2))
            
            # Generate flight number
            flight_id = f"FL{date.strftime('%m%d')}{i:04d}"
            
            flights.append({
                'flight_id': flight_id,
                'aircraft_type': aircraft_type,
                'arrival_time': arrival_time,
                'scheduled_departure': scheduled_departure,
                'position': position,
                'passenger_count': self.aircraft_types[aircraft_type].pax_capacity,
                'is_international': np.random.choice([True, False], p=[0.6, 0.4])
            })
        
        df = pd.DataFrame(flights)
        df = df.sort_values('arrival_time').reset_index(drop=True)
        
        return df
    
    def _generate_arrival_time(self, date: datetime) -> datetime:
        """Generate realistic arrival time with peak hour weighting"""
        peak_hours = self.gen_config['generation']['operational_parameters']['peak_hours'] \
            if 'operational_parameters' in self.gen_config['generation'] \
            else [7, 8, 13, 18, 19, 20]
        
        # Operating hours
        start_hour = 6
        end_hour = 23
        
        # Create weighted hour distribution
        hours = list(range(start_hour, end_hour))
        weights = [3 if h in peak_hours else 1 for h in hours]
        weights = np.array(weights) / sum(weights)
        
        hour = np.random.choice(hours, p=weights)
        minute = np.random.randint(0, 60)
        second = np.random.randint(0, 60)
        
        arrival_time = datetime.combine(date.date(), datetime.min.time())
        arrival_time = arrival_time.replace(hour=hour, minute=minute, second=second)
        
        return arrival_time
    
    def add_stochastic_delays(self, schedule: pd.DataFrame) -> pd.DataFrame:
        """
        Add realistic delays to the schedule
        
        Args:
            schedule: DataFrame with flight schedule
        
        Returns:
            Modified DataFrame with actual arrival times and delay factors
        """
        delay_config = self.gen_config['generation']['delays']
        
        # Arrival delays
        if delay_config['arrival_delay']['enabled']:
            if delay_config['arrival_delay']['distribution'] == 'gamma':
                params = delay_config['arrival_delay']['params']
                delays = np.random.gamma(
                    shape=params['shape'],
                    scale=params['scale'],
                    size=len(schedule)
                )
                # Cap maximum delay
                max_delay = delay_config['arrival_delay']['max_delay_minutes']
                delays = np.minimum(delays, max_delay)
            
            schedule['arrival_delay_minutes'] = delays
            schedule['actual_arrival'] = schedule['arrival_time'] + pd.to_timedelta(
                schedule['arrival_delay_minutes'], unit='m'
            )
        else:
            schedule['arrival_delay_minutes'] = 0
            schedule['actual_arrival'] = schedule['arrival_time']
        
        # Equipment failures
        if delay_config['equipment_failure']['enabled']:
            failure_prob = delay_config['equipment_failure']['probability']
            schedule['equipment_failure'] = np.random.choice(
                [True, False],
                size=len(schedule),
                p=[failure_prob, 1 - failure_prob]
            )
            
            # Randomly select affected task for failures
            affected_tasks = delay_config['equipment_failure']['affected_tasks']
            schedule['failed_task'] = schedule.apply(
                lambda row: np.random.choice(affected_tasks) if row['equipment_failure'] else None,
                axis=1
            )
        else:
            schedule['equipment_failure'] = False
            schedule['failed_task'] = None
        
        return schedule
    
    def generate_task_list(
    self,
    flight_row: pd.Series,
    weather_multiplier: float = 1.0
) -> List[Dict]:
        """
        Generate detailed task list for a specific flight
        
        Args:
            flight_row: Single row from flight schedule DataFrame
            weather_multiplier: Factor to multiply task durations (for weather impact)
        
        Returns:
            List of task dictionaries
        """
        aircraft_type = self.aircraft_types[flight_row['aircraft_type']]
        tasks = []
        
        task_variance = self.gen_config['generation']['delays']['task_duration_variance']['factor']
        
        for task_name, task_obj in aircraft_type.tasks.items():
            # Get duration with variance
            base_duration = task_obj.duration
            duration = int(np.random.uniform(
                base_duration * (1 - task_variance),
                base_duration * (1 + task_variance)
            ))
            
            # Apply weather multiplier
            duration = int(duration * weather_multiplier)
            
            # Apply equipment failure delay (if this task is affected)
            if flight_row['equipment_failure'] and flight_row['failed_task'] == task_name:
                duration = int(duration * 1.5)  # 50% longer due to equipment issue
            
            # Ensure required_vehicles is a list
            required_vehicles = task_obj.required_vehicles
            if isinstance(required_vehicles, str):
                required_vehicles = [required_vehicles]
            elif not isinstance(required_vehicles, list):
                required_vehicles = []
            
            tasks.append({
                'flight_id': flight_row['flight_id'],
                'task_name': task_name,
                'duration': duration,
                'required_vehicles': required_vehicles,
                'predecessors': aircraft_type.precedence[task_name],
                'earliest_start': None,  # Will be calculated during scheduling
                'actual_start': None,
                'actual_end': None
            })
        
        return tasks
    
    def save_aircraft_specs(self, output_path: str):
        """Save aircraft specifications to JSON"""
        import json
        
        specs = {}
        for type_code, aircraft in self.aircraft_types.items():
            specs[type_code] = {
                'category': aircraft.category,
                'pax_capacity': aircraft.pax_capacity,
                'turnaround_time': aircraft.turnaround_time,
                'tasks': {
                    name: {
                        'duration': task.duration,
                        'required_vehicles': task.required_vehicles
                    }
                    for name, task in aircraft.tasks.items()
                },
                'precedence': aircraft.precedence
            }
        
        with open(output_path, 'w') as f:
            json.dump(specs, f, indent=2)
        
        print(f"✅ Aircraft specifications saved to {output_path}")