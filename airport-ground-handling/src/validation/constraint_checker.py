"""
Validate generated dataset against constraints
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


class DatasetValidator:
    """Validates generated dataset for consistency and realism"""
    
    def __init__(self, flights_df: pd.DataFrame, tasks_df: pd.DataFrame):
        self.flights = flights_df
        self.tasks = tasks_df
        self.validation_results = []
    
    def validate_all(self) -> Dict:
        """Run all validation checks"""
        print("🔍 Running validation checks...\n")
        
        checks = [
            ("Temporal Consistency", self.check_temporal_consistency),
            ("Task Completeness", self.check_task_completeness),
            ("Duration Reasonableness", self.check_duration_reasonableness),
            ("Precedence Validity", self.check_precedence_validity),
            ("Position Validity", self.check_position_validity),
            ("Delay Distribution", self.check_delay_distribution),
        ]
        
        results = {}
        
        for check_name, check_func in checks:
            print(f"Running: {check_name}...")
            result = check_func()
            results[check_name] = result
            
            if result['passed']:
                print(f"  ✅ PASSED")
            else:
                print(f"  ❌ FAILED: {result['message']}")
            print()
        
        # Overall summary
        passed = sum(1 for r in results.values() if r['passed'])
        total = len(results)
        
        print("=" * 60)
        print(f"Validation Summary: {passed}/{total} checks passed")
        print("=" * 60)
        
        return results
    
    def check_temporal_consistency(self) -> Dict:
        """Check that times are logically consistent"""
        errors = []
        
        # Arrival before departure
        invalid = self.flights[
            self.flights['actual_arrival'] >= self.flights['scheduled_departure']
        ]
        
        if len(invalid) > 0:
            errors.append(f"{len(invalid)} flights have arrival >= departure")
        
        # Scheduled departure after arrival
        invalid = self.flights[
            self.flights['scheduled_departure'] <= self.flights['arrival_time']
        ]
        
        if len(invalid) > 0:
            errors.append(f"{len(invalid)} flights have scheduled departure before/at arrival")
        
        # Check times are within operating hours (6-23)
        self.flights['arrival_hour'] = pd.to_datetime(self.flights['actual_arrival']).dt.hour
        invalid_hours = self.flights[
            (self.flights['arrival_hour'] < 6) | (self.flights['arrival_hour'] > 23)
        ]
        
        if len(invalid_hours) > 0:
            errors.append(f"{len(invalid_hours)} flights outside operating hours")
        
        return {
            'passed': len(errors) == 0,
            'message': '; '.join(errors) if errors else 'All temporal checks passed'
        }
    
    def check_task_completeness(self) -> Dict:
        """Check that each flight has all required tasks"""
        errors = []
        
        # Expected tasks per aircraft type (simplified check)
        expected_tasks = {
            'A320': 9, 'B737': 9, 'A321': 9,
            'B777': 10, 'A350': 10
        }
        
        task_counts = self.tasks.groupby('flight_id').size()
        
        for flight_id, row in self.flights.iterrows():
            expected = expected_tasks.get(row['aircraft_type'], 9)
            actual = task_counts.get(row['flight_id'], 0)
            
            if actual != expected:
                errors.append(
                    f"{row['flight_id']} has {actual} tasks, expected {expected}"
                )
        
        # Limit error messages
        if len(errors) > 10:
            errors = errors[:10] + [f"... and {len(errors) - 10} more"]
        
        return {
            'passed': len(errors) == 0,
            'message': '; '.join(errors) if errors else 'All flights have complete tasks',
            'error_count': len(errors)
        }
    
    def check_duration_reasonableness(self) -> Dict:
        """Check task durations are within reasonable bounds"""
        errors = []
        
        # Define reasonable bounds (min, max) in minutes
        duration_bounds = {
            'deplaning': (5, 20),
            'boarding': (10, 35),
            'refueling': (8, 30),
            'catering': (10, 40),
            'cleaning': (15, 50),
            'cargo_unload': (8, 35),
            'cargo_load': (8, 35),
            'water_service': (5, 15),
            'lavatory_service': (5, 15),
            'pushback': (3, 10)
        }
        
        for task_name, (min_dur, max_dur) in duration_bounds.items():
            task_subset = self.tasks[self.tasks['task_name'] == task_name]
            
            if len(task_subset) == 0:
                continue
            
            invalid = task_subset[
                (task_subset['duration'] < min_dur) | 
                (task_subset['duration'] > max_dur)
            ]
            
            if len(invalid) > 0:
                errors.append(
                    f"{len(invalid)} {task_name} tasks outside bounds [{min_dur}, {max_dur}]"
                )
        
        return {
            'passed': len(errors) == 0,
            'message': '; '.join(errors) if errors else 'All durations reasonable'
        }
    
    def check_precedence_validity(self) -> Dict:
        """Verify precedence constraints are respected"""
        # This is a simplified check - full validation would require
        # actual scheduling information
        
        # Check that tasks have valid predecessors
        valid_tasks = set([
            'deplaning', 'boarding', 'refueling', 'catering', 'cleaning',
            'cargo_unload', 'cargo_load', 'water_service', 
            'lavatory_service', 'pushback'
        ])
        
        invalid_predecessors = []
        
        for task in self.tasks.itertuples():
            if not isinstance(task.predecessors, list):
                continue
            
            for pred in task.predecessors:
                if pred not in valid_tasks:
                    invalid_predecessors.append(
                        f"{task.flight_id}: {task.task_name} has invalid predecessor {pred}"
                    )
        
        return {
            'passed': len(invalid_predecessors) == 0,
            'message': 'Precedence structure valid' if len(invalid_predecessors) == 0 
                      else f"{len(invalid_predecessors)} invalid predecessors found"
        }
    
    def check_position_validity(self) -> Dict:
        """Check that assigned positions exist"""
        # This would need the airport layout data
        # For now, just check format
        
        valid_patterns = ['gate_', 'remote_stand_', 'terminal_']
        
        invalid_positions = self.flights[
            ~self.flights['position'].str.contains('|'.join(valid_patterns), na=False)
        ]
        
        return {
            'passed': len(invalid_positions) == 0,
            'message': 'All positions valid' if len(invalid_positions) == 0
                      else f"{len(invalid_positions)} flights have invalid positions"
        }
    
    def check_delay_distribution(self) -> Dict:
        """Check that delays follow expected distribution"""
        delays = self.flights['arrival_delay_minutes']
        
        issues = []
        
        # Check for negative delays (shouldn't exist)
        if (delays < 0).any():
            issues.append(f"{(delays < 0).sum()} flights have negative delays")
        
        # Check mean is reasonable (5-15 minutes typical)
        mean_delay = delays.mean()
        if mean_delay < 2 or mean_delay > 30:
            issues.append(f"Mean delay {mean_delay:.1f} min seems unusual")
        
        # Check distribution is right-skewed (most flights small delay, few large)
        median_delay = delays.median()
        if mean_delay < median_delay:  # Should be mean > median for right skew
            issues.append("Delay distribution not right-skewed as expected")
        
        return {
            'passed': len(issues) == 0,
            'message': '; '.join(issues) if issues else 'Delay distribution looks realistic',
            'mean_delay': mean_delay,
            'median_delay': median_delay
        }


def validate_dataset(
    flights_path: str = "data/raw/flight_schedules.csv",
    tasks_path: str = "data/raw/tasks.csv"
) -> Dict:
    """
    Main validation function
    
    Args:
        flights_path: Path to flights CSV
        tasks_path: Path to tasks CSV
    
    Returns:
        Dictionary with validation results
    """
    print("📂 Loading dataset...")
    flights_df = pd.read_csv(flights_path)
    tasks_df = pd.read_csv(tasks_path)
    
    print(f"   Flights: {len(flights_df)}")
    print(f"   Tasks: {len(tasks_df)}\n")
    
    validator = DatasetValidator(flights_df, tasks_df)
    results = validator.validate_all()
    
    return results