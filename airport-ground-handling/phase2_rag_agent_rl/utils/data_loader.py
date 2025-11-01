"""
Data loading utilities for Phase 2
"""

import pandas as pd
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class DatasetLoader:
    """Load and preprocess generated dataset for Phase 2"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        
        # Load main datasets
        self.flights_df = pd.read_csv(self.data_dir / "flight_schedules.csv")
        self.tasks_df = pd.read_csv(self.data_dir / "tasks.csv")
        
        # Convert datetime columns
        datetime_cols = ['arrival_time', 'actual_arrival', 'scheduled_departure']
        for col in datetime_cols:
            if col in self.flights_df.columns:
                self.flights_df[col] = pd.to_datetime(self.flights_df[col])
        
        # Load configurations
        with open(self.data_dir / "aircraft_specs.json", 'r') as f:
            self.aircraft_specs = json.load(f)
        
        with open(self.data_dir / "vehicle_fleet.json", 'r') as f:
            self.vehicle_fleet = json.load(f)
        
        with open(self.data_dir / "airport_layout.json", 'r') as f:
            self.airport_layout = json.load(f)
        
        self.travel_times = pd.read_csv(
            self.data_dir / "travel_times.csv", 
            index_col=0
        )
        
        print(f"✅ Loaded {len(self.flights_df)} flights and {len(self.tasks_df)} tasks")
    
    def get_scenario_by_date(self, date: str) -> Dict:
        """Get all flights and tasks for a specific date"""
        date_flights = self.flights_df[self.flights_df['date'] == date].copy()
        flight_ids = date_flights['flight_id'].tolist()
        date_tasks = self.tasks_df[self.tasks_df['flight_id'].isin(flight_ids)].copy()
        
        return {
            'date': date,
            'flights': date_flights,
            'tasks': date_tasks,
            'num_flights': len(date_flights),
            'num_tasks': len(date_tasks)
        }
    
    def get_scenario_by_time_window(
        self, 
        start_time: str, 
        end_time: str
    ) -> Dict:
        """Get flights arriving in a time window"""
        mask = (
            (self.flights_df['actual_arrival'] >= start_time) & 
            (self.flights_df['actual_arrival'] <= end_time)
        )
        window_flights = self.flights_df[mask].copy()
        flight_ids = window_flights['flight_id'].tolist()
        window_tasks = self.tasks_df[self.tasks_df['flight_id'].isin(flight_ids)].copy()
        
        return {
            'start_time': start_time,
            'end_time': end_time,
            'flights': window_flights,
            'tasks': window_tasks,
            'num_flights': len(window_flights),
            'num_tasks': len(window_tasks)
        }
    
    def load_scenario_files(
        self, 
        scenarios_dir: str = "data/processed/scenarios"
    ) -> List[Dict]:
        """Load pre-generated scenario files"""
        scenarios_path = Path(scenarios_dir)
        scenario_files = sorted(scenarios_path.glob("scenario_*.json"))
        
        scenarios = []
        for file in scenario_files:
            with open(file, 'r') as f:
                scenario = json.load(f)
                scenarios.append(scenario)
        
        print(f"✅ Loaded {len(scenarios)} scenario files")
        return scenarios
    
    def get_random_scenarios(self, n: int = 10) -> List[Dict]:
        """Get n random scenario chunks"""
        all_scenarios = self.load_scenario_files()
        if n > len(all_scenarios):
            n = len(all_scenarios)
        
        indices = np.random.choice(len(all_scenarios), n, replace=False)
        return [all_scenarios[i] for i in indices]
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        return {
            'total_flights': len(self.flights_df),
            'total_tasks': len(self.tasks_df),
            'date_range': {
                'start': self.flights_df['date'].min(),
                'end': self.flights_df['date'].max()
            },
            'aircraft_types': self.flights_df['aircraft_type'].value_counts().to_dict(),
            'avg_delay': self.flights_df['arrival_delay_minutes'].mean(),
            'equipment_failures': self.flights_df['equipment_failure'].sum()
        }


class ScenarioConverter:
    """Convert scenarios to different formats for RAG/RL"""
    
    @staticmethod
    def to_text_description(scenario: Dict) -> str:
        """Convert scenario to natural language description"""
        
        text = f"""
Airport Operations Scenario

Date: {scenario.get('date', 'Unknown')}
Number of Flights: {scenario['num_flights']}
Time Window: {scenario.get('time_window', {}).get('start', 'N/A')} to {scenario.get('time_window', {}).get('end', 'N/A')}

Aircraft Mix:
"""
        
        if 'aircraft_mix' in scenario:
            for aircraft, count in scenario['aircraft_mix'].items():
                text += f"  - {aircraft}: {count} flights\n"
        
        text += f"\nTotal Tasks: {scenario.get('statistics', {}).get('total_tasks', 'N/A')}\n"
        text += f"Average Delay: {scenario.get('statistics', {}).get('avg_delay', 0):.2f} minutes\n"
        text += f"Equipment Failures: {scenario.get('statistics', {}).get('equipment_failures', 0)}\n"
        
        return text
    
    @staticmethod
    def to_rl_state(scenario: Dict, current_time: float = 0) -> Dict:
        """Convert scenario to RL environment state"""
        
        flights_df = pd.DataFrame(scenario['flights'])
        tasks_df = pd.DataFrame(scenario['tasks'])
        
        # Normalize times
        if 'actual_arrival' in flights_df.columns:
            flights_df['actual_arrival'] = pd.to_datetime(flights_df['actual_arrival'])
            min_time = flights_df['actual_arrival'].min()
            flights_df['time_offset'] = (
                flights_df['actual_arrival'] - min_time
            ).dt.total_seconds() / 60
        
        state = {
            'num_flights': len(flights_df),
            'num_tasks': len(tasks_df),
            'aircraft_types': flights_df['aircraft_type'].value_counts().to_dict(),
            'current_time': current_time,
            'flights': flights_df.to_dict('records'),
            'tasks': tasks_df.to_dict('records')
        }
        
        return state