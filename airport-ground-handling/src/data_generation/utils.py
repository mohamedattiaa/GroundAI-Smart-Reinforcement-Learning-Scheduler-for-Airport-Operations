"""
Utility functions for data generation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple


def calculate_earliest_start_times(
    tasks: List[Dict],
    arrival_time: datetime
) -> List[Dict]:
    """
    Calculate earliest possible start time for each task based on precedence
    
    Args:
        tasks: List of task dictionaries
        arrival_time: Aircraft arrival time
    
    Returns:
        Updated task list with earliest_start times
    """
    # Create task lookup
    task_dict = {task['task_name']: task for task in tasks}
    
    # Calculate earliest start recursively
    def calc_earliest(task_name: str, current_time: datetime) -> datetime:
        task = task_dict[task_name]
        
        if task['earliest_start'] is not None:
            return task['earliest_start']
        
        # Base case: no predecessors
        if not task['predecessors']:
            task['earliest_start'] = current_time
            return current_time
        
        # Recursive case: must wait for all predecessors
        max_pred_end = current_time
        for pred_name in task['predecessors']:
            pred_start = calc_earliest(pred_name, current_time)
            pred_end = pred_start + timedelta(minutes=task_dict[pred_name]['duration'])
            max_pred_end = max(max_pred_end, pred_end)
        
        task['earliest_start'] = max_pred_end
        return max_pred_end
    
    # Calculate for all tasks
    for task in tasks:
        calc_earliest(task['task_name'], arrival_time)
    
    return tasks


def validate_precedence_constraints(tasks: List[Dict]) -> bool:
    """
    Check if all precedence constraints are satisfied
    
    Args:
        tasks: List of tasks with actual_start and actual_end times
    
    Returns:
        True if all constraints satisfied, False otherwise
    """
    task_dict = {task['task_name']: task for task in tasks}
    
    for task in tasks:
        if task['actual_start'] is None:
            continue
        
        for pred_name in task['predecessors']:
            pred = task_dict[pred_name]
            
            if pred['actual_end'] is None:
                return False
            
            if task['actual_start'] < pred['actual_end']:
                print(f"❌ Precedence violation: {task['task_name']} started before {pred_name} finished")
                return False
    
    return True


def calculate_critical_path(tasks: List[Dict]) -> Tuple[List[str], float]:
    """
    Find the critical path (longest path) through the task network
    
    Args:
        tasks: List of task dictionaries with durations and precedence
    
    Returns:
        Tuple of (critical_path_tasks, total_duration)
    """
    task_dict = {task['task_name']: task for task in tasks}
    
    # Calculate longest path to each task
    longest_path = {}
    predecessors_on_path = {}
    
    def calc_longest_path(task_name: str) -> float:
        if task_name in longest_path:
            return longest_path[task_name]
        
        task = task_dict[task_name]
        
        if not task['predecessors']:
            longest_path[task_name] = task['duration']
            predecessors_on_path[task_name] = []
            return task['duration']
        
        max_path = 0
        best_pred = None
        
        for pred_name in task['predecessors']:
            pred_path = calc_longest_path(pred_name)
            if pred_path > max_path:
                max_path = pred_path
                best_pred = pred_name
        
        longest_path[task_name] = max_path + task['duration']
        predecessors_on_path[task_name] = predecessors_on_path[best_pred] + [best_pred]
        
        return longest_path[task_name]
    
    # Find task with longest path (usually the final task)
    max_duration = 0
    final_task = None
    
    for task in tasks:
        duration = calc_longest_path(task['task_name'])
        if duration > max_duration:
            max_duration = duration
            final_task = task['task_name']
    
    # Reconstruct critical path
    critical_path = predecessors_on_path[final_task] + [final_task]
    
    return critical_path, max_duration


def format_duration(minutes: float) -> str:
    """Format duration in minutes to human-readable string"""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    
    if hours > 0:
        return f"{hours}h {mins}m"
    else:
        return f"{mins}m"


def calculate_aircraft_turnaround(tasks: List[Dict]) -> Dict:
    """
    Calculate turnaround statistics for a flight
    
    Args:
        tasks: List of completed tasks
    
    Returns:
        Dictionary with turnaround statistics
    """
    if not tasks or not all(task['actual_end'] for task in tasks):
        return None
    
    arrival = min(task['actual_start'] for task in tasks if task['actual_start'])
    departure = max(task['actual_end'] for task in tasks if task['actual_end'])
    
    turnaround_time = (departure - arrival).total_seconds() / 60
    
    critical_path, critical_duration = calculate_critical_path(tasks)
    
    idle_time = turnaround_time - critical_duration
    efficiency = (critical_duration / turnaround_time * 100) if turnaround_time > 0 else 0
    
    return {
        'arrival_time': arrival,
        'departure_time': departure,
        'turnaround_minutes': turnaround_time,
        'critical_path': critical_path,
        'critical_path_duration': critical_duration,
        'idle_time': idle_time,
        'efficiency_percent': efficiency
    }