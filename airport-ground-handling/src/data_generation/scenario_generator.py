"""
Main scenario generation orchestrator
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

from .airport_config import AirportConfig
from .aircraft_generator import AircraftGenerator
from .vehicle_generator import VehicleFleet


class ScenarioGenerator:
    """
    Orchestrates the generation of complete airport operation scenarios
    """
    
    def __init__(
        self,
        airport_config_path: str = "configs/airport_config.yaml",
        aircraft_config_path: str = "configs/aircraft_types.yaml",
        generation_config_path: str = "configs/generation_config.yaml"
    ):
        print("🚀 Initializing Scenario Generator...")
        
        self.airport = AirportConfig(airport_config_path)
        self.aircraft_gen = AircraftGenerator(aircraft_config_path, generation_config_path)
        self.fleet = VehicleFleet(airport_config_path)
        
        # Load generation config
        self.gen_config = self.aircraft_gen.gen_config['generation']
        
        print(f"✅ Airport: {self.airport.name} ({self.airport.code})")
        print(f"✅ Terminals: {len(self.airport.terminals)}")
        print(f"✅ Total positions: {len(self.airport.all_positions)}")
        print(f"✅ Vehicle fleet size: {len(self.fleet.get_all_vehicles())}")
    
    def generate_full_dataset(
        self,
        output_dir: str = "data/raw",
        num_days: int = None
    ) -> Dict[str, str]:
        """
        Generate complete multi-day dataset
        
        Args:
            output_dir: Directory to save generated files
            num_days: Number of days to generate (if None, uses config)
        
        Returns:
            Dictionary with paths to generated files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if num_days is None:
            num_days = self.gen_config['num_days']
        
        start_date = datetime.strptime(
            self.gen_config['start_date'],
            "%Y-%m-%d"
        )
        
        print(f"\n📅 Generating {num_days} days of data starting from {start_date.date()}")
        
        all_flights = []
        all_tasks = []
        
        # Generate day by day
        for day in tqdm(range(num_days), desc="Generating days"):
            current_date = start_date + timedelta(days=day)
            
            # Check for weather impact
            weather_multiplier = self._check_weather_impact(current_date)
            
            # Generate daily schedule
            daily_schedule = self.aircraft_gen.generate_daily_schedule(
                date=current_date,
                airport_config=self.airport
            )
            
            # Add stochastic delays
            daily_schedule = self.aircraft_gen.add_stochastic_delays(daily_schedule)
            
            # Add day identifier
            daily_schedule['date'] = current_date.date()
            daily_schedule['day_of_week'] = current_date.strftime('%A')
            daily_schedule['is_weekend'] = current_date.weekday() >= 5
            
            all_flights.append(daily_schedule)
            
            # Generate tasks for each flight
            for _, flight in daily_schedule.iterrows():
                tasks = self.aircraft_gen.generate_task_list(
                    flight,
                    weather_multiplier=weather_multiplier
                )
                all_tasks.extend(tasks)
        
        # Combine all data
        flights_df = pd.concat(all_flights, ignore_index=True)
        tasks_df = pd.DataFrame(all_tasks)
        
        # Save datasets
        output_files = {}
        
        # 1. Flight schedules
        flights_path = output_path / "flight_schedules.csv"
        flights_df.to_csv(flights_path, index=False)
        output_files['flights'] = str(flights_path)
        print(f"✅ Saved flight schedules: {flights_path}")
        
        # 2. Tasks
        tasks_path = output_path / "tasks.csv"
        tasks_df.to_csv(tasks_path, index=False)
        output_files['tasks'] = str(tasks_path)
        print(f"✅ Saved tasks: {tasks_path}")
        
        # 3. Airport layout
        layout_path = output_path / "airport_layout.json"
        self.airport.save_layout(layout_path)
        output_files['layout'] = str(layout_path)
        
        # 4. Aircraft specs
        aircraft_path = output_path / "aircraft_specs.json"
        self.aircraft_gen.save_aircraft_specs(aircraft_path)
        output_files['aircraft'] = str(aircraft_path)
        
        # 5. Vehicle fleet
        fleet_path = output_path / "vehicle_fleet.json"
        self.fleet.save_fleet_config(fleet_path)
        output_files['fleet'] = str(fleet_path)
        
        # 6. Travel times
        travel_path = output_path / "travel_times.csv"
        positions_list = list(self.airport.all_positions.keys())
        travel_df = pd.DataFrame(
            self.airport.travel_time_matrix,
            index=positions_list,
            columns=positions_list
        )
        travel_df.to_csv(travel_path)
        output_files['travel_times'] = str(travel_path)
        print(f"✅ Saved travel times: {travel_path}")
        
        # 7. Generate statistics
        stats = self._generate_statistics(flights_df, tasks_df)
        stats_path = output_path.parent / "statistics" / "dataset_summary.txt"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(stats_path, 'w') as f:
            f.write(stats)
        output_files['statistics'] = str(stats_path)
        print(f"✅ Saved statistics: {stats_path}")
        
        print(f"\n🎉 Dataset generation complete!")
        print(f"📊 Total flights: {len(flights_df)}")
        print(f"📊 Total tasks: {len(tasks_df)}")
        print(f"📁 Output directory: {output_path}")
        
        return output_files
    
    def _check_weather_impact(self, date: datetime) -> float:
        """Determine if weather impacts operations on this day"""
        weather_config = self.gen_config['delays']['weather_impact']
        
        if not weather_config['enabled']:
            return 1.0
        
        # Random weather event
        has_weather_issue = np.random.random() < weather_config['probability']
        
        if has_weather_issue:
            return weather_config['delay_multiplier']
        else:
            return 1.0
    
    def _generate_statistics(
        self,
        flights_df: pd.DataFrame,
        tasks_df: pd.DataFrame
    ) -> str:
        """Generate summary statistics"""
        
        stats = []
        stats.append("=" * 60)
        stats.append("AIRPORT GROUND HANDLING DATASET STATISTICS")
        stats.append("=" * 60)
        stats.append("")
        
        # Flight statistics
        stats.append("FLIGHT STATISTICS")
        stats.append("-" * 60)
        stats.append(f"Total flights: {len(flights_df)}")
        stats.append(f"Date range: {flights_df['date'].min()} to {flights_df['date'].max()}")
        stats.append(f"Number of days: {flights_df['date'].nunique()}")
        stats.append("")
        
        stats.append("Daily flight distribution:")
        daily_stats = flights_df.groupby('date').size()
        stats.append(f"  Average: {daily_stats.mean():.1f}")
        stats.append(f"  Min: {daily_stats.min()}")
        stats.append(f"  Max: {daily_stats.max()}")
        stats.append(f"  Std Dev: {daily_stats.std():.1f}")
        stats.append("")
        
        stats.append("Weekday vs Weekend:")
        weekend_stats = flights_df.groupby('is_weekend').size()
        stats.append(f"  Weekday flights: {weekend_stats.get(False, 0)} ({weekend_stats.get(False, 0)/len(flights_df)*100:.1f}%)")
        stats.append(f"  Weekend flights: {weekend_stats.get(True, 0)} ({weekend_stats.get(True, 0)/len(flights_df)*100:.1f}%)")
        stats.append("")
        
        # Aircraft mix
        stats.append("Aircraft type distribution:")
        aircraft_dist = flights_df['aircraft_type'].value_counts()
        for ac_type, count in aircraft_dist.items():
            stats.append(f"  {ac_type}: {count} ({count/len(flights_df)*100:.1f}%)")
        stats.append("")
        
        # Delays
        stats.append("DELAY STATISTICS")
        stats.append("-" * 60)
        stats.append(f"Average arrival delay: {flights_df['arrival_delay_minutes'].mean():.2f} minutes")
        stats.append(f"Median arrival delay: {flights_df['arrival_delay_minutes'].median():.2f} minutes")
        stats.append(f"Max arrival delay: {flights_df['arrival_delay_minutes'].max():.2f} minutes")
        stats.append(f"Flights with delay > 15 min: {(flights_df['arrival_delay_minutes'] > 15).sum()} ({(flights_df['arrival_delay_minutes'] > 15).sum()/len(flights_df)*100:.1f}%)")
        stats.append("")
        
        # Equipment failures
        stats.append("Equipment failures:")
        failures = flights_df['equipment_failure'].sum()
        stats.append(f"  Total: {failures} ({failures/len(flights_df)*100:.1f}%)")
        stats.append("")
        
        # Task statistics
        stats.append("TASK STATISTICS")
        stats.append("-" * 60)
        stats.append(f"Total tasks: {len(tasks_df)}")
        stats.append(f"Average tasks per flight: {len(tasks_df)/len(flights_df):.1f}")
        stats.append("")
        
        stats.append("Task distribution:")
        task_dist = tasks_df['task_name'].value_counts()
        for task, count in task_dist.items():
            stats.append(f"  {task}: {count}")
        stats.append("")
        
        stats.append("Average task duration (minutes):")
        avg_durations = tasks_df.groupby('task_name')['duration'].mean().sort_values(ascending=False)
        for task, duration in avg_durations.items():
            stats.append(f"  {task}: {duration:.1f}")
        stats.append("")
        
        # Position statistics
        stats.append("POSITION UTILIZATION")
        stats.append("-" * 60)
        position_dist = flights_df['position'].value_counts().head(10)
        stats.append("Top 10 most used positions:")
        for pos, count in position_dist.items():
            stats.append(f"  {pos}: {count} flights")
        stats.append("")
        
        stats.append("=" * 60)
        
        return "\n".join(stats)
    
    def generate_scenario_chunks(
        self,
        flights_df: pd.DataFrame,
        tasks_df: pd.DataFrame,
        output_dir: str = "data/processed/scenarios",
        chunk_size: int = 100
    ):
        """
        Split dataset into scenario chunks for RAG/RL training
        
        Args:
            flights_df: DataFrame with all flights
            tasks_df: DataFrame with all tasks
            output_dir: Directory to save scenario files
            chunk_size: Number of flights per scenario
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\n📦 Generating scenario chunks (size: {chunk_size} flights)...")
        
        # Convert datetime columns to proper datetime objects
        datetime_columns = ['arrival_time', 'actual_arrival', 'scheduled_departure']
        for col in datetime_columns:
            if col in flights_df.columns:
                flights_df[col] = pd.to_datetime(flights_df[col])
        
        # Replace NaN with None (which becomes null in JSON)
        flights_df = flights_df.replace({np.nan: None})
        tasks_df = tasks_df.replace({np.nan: None})
        
        # Sort by arrival time
        flights_df = flights_df.sort_values('actual_arrival').reset_index(drop=True)
        
        num_scenarios = len(flights_df) // chunk_size + (1 if len(flights_df) % chunk_size > 0 else 0)
        
        for i in tqdm(range(num_scenarios), desc="Creating scenarios"):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(flights_df))
            
            scenario_flights = flights_df.iloc[start_idx:end_idx].copy()
            flight_ids = scenario_flights['flight_id'].tolist()
            
            # Get tasks for these flights
            scenario_tasks = tasks_df[tasks_df['flight_id'].isin(flight_ids)].copy()
            
            # Convert datetime objects to strings for JSON serialization
            for col in datetime_columns:
                if col in scenario_flights.columns:
                    scenario_flights[col] = scenario_flights[col].astype(str)
            
            # Create scenario object
            scenario = {
                'scenario_id': f"scenario_{i+1:04d}",
                'num_flights': len(scenario_flights),
                'time_window': {
                    'start': str(scenario_flights['actual_arrival'].min()),
                    'end': str(scenario_flights['scheduled_departure'].max())
                },
                'aircraft_mix': scenario_flights['aircraft_type'].value_counts().to_dict(),
                'flights': scenario_flights.to_dict('records'),
                'tasks': scenario_tasks.to_dict('records'),
                'statistics': {
                    'total_tasks': len(scenario_tasks),
                    'avg_delay': float(scenario_flights['arrival_delay_minutes'].mean()),
                    'equipment_failures': int(scenario_flights['equipment_failure'].sum())
                }
            }
            
            # Save scenario with proper null handling
            scenario_path = output_path / f"scenario_{i+1:04d}.json"
            with open(scenario_path, 'w') as f:
                json.dump(scenario, f, indent=2, default=str, allow_nan=False)
        
        print(f"✅ Generated {num_scenarios} scenario files in {output_path}")