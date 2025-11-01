"""
Aircraft Agent - Represents an aircraft requiring ground services
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class AircraftState:
    """State information for an aircraft"""
    flight_id: str
    aircraft_type: str
    position: str
    arrival_time: pd.Timestamp
    scheduled_departure: pd.Timestamp
    tasks: List[Dict]
    tasks_completed: List[str]
    current_delay: float
    priority_score: float


class AircraftAgent:
    """
    Agent representing an aircraft
    Responsible for requesting services and monitoring turnaround
    """
    
    def __init__(
        self,
        flight_id: str,
        aircraft_type: str,
        position: str,
        arrival_time: pd.Timestamp,
        scheduled_departure: pd.Timestamp,
        tasks: List[Dict]
    ):
        self.flight_id = flight_id
        self.aircraft_type = aircraft_type
        self.position = position
        self.arrival_time = arrival_time
        self.scheduled_departure = scheduled_departure
        self.tasks = tasks
        
        # State tracking
        self.tasks_completed = []
        self.tasks_in_progress = []
        self.current_delay = 0.0
        self.priority_score = self._calculate_priority()
        
        # Assigned vehicles
        self.assigned_vehicles = {}
    
    def _calculate_priority(self) -> float:
        """Calculate priority score for this aircraft"""
        
        # Factors affecting priority:
        # 1. Time until departure (higher priority if closer)
        # 2. Aircraft size (wide-body gets priority)
        # 3. International vs domestic
        
        priority = 1.0
        
        # Wide-body aircraft get higher priority
        if self.aircraft_type in ['B777', 'A350', 'B787', 'A330']:
            priority += 0.3
        
        # Critical path length (more tasks = higher priority)
        priority += len(self.tasks) * 0.05
        
        return priority
    
    def get_next_tasks(self) -> List[Dict]:
        """Get tasks that are ready to start"""
        
        ready_tasks = []
        
        for task in self.tasks:
            task_name = task['task_name']
            
            # Skip if already completed or in progress
            if task_name in self.tasks_completed:
                continue
            if task_name in self.tasks_in_progress:
                continue
            
            # Check if all predecessors are completed
            predecessors = task.get('predecessors', [])
            
            # Handle both string and list formats for predecessors
            if isinstance(predecessors, str):
                # Convert comma-separated string to list
                predecessors = [p.strip() for p in predecessors.split(',') if p.strip()]
            elif not isinstance(predecessors, list):
                predecessors = []
            
            if all(pred in self.tasks_completed for pred in predecessors):
                ready_tasks.append(task)
        
        return ready_tasks
    
    def request_service(self, task: Dict) -> Dict:
        """Create a service request for a task"""
        
        request = {
            'flight_id': self.flight_id,
            'aircraft_type': self.aircraft_type,
            'position': self.position,
            'task_name': task['task_name'],
            'duration': task['duration'],
            'required_vehicles': task['required_vehicles'],
            'priority': self.priority_score,
            'urgency': self._calculate_urgency(task)
        }
        
        return request
    
    def _calculate_urgency(self, task: Dict) -> float:
        """Calculate urgency for a specific task"""
        
        # Calculate time remaining until departure
        time_buffer = (self.scheduled_departure - pd.Timestamp.now()).total_seconds() / 60
        
        # If we're running late, increase urgency
        if time_buffer < 30:
            return 1.0  # Critical
        elif time_buffer < 60:
            return 0.7  # High
        elif time_buffer < 90:
            return 0.5  # Medium
        else:
            return 0.3  # Normal
    
    def assign_vehicle(self, task_name: str, vehicle_id: str):
        """Record vehicle assignment"""
        self.assigned_vehicles[task_name] = vehicle_id
        if task_name not in self.tasks_in_progress:
            self.tasks_in_progress.append(task_name)
    
    def complete_task(self, task_name: str):
        """Mark a task as completed"""
        if task_name in self.tasks_in_progress:
            self.tasks_in_progress.remove(task_name)
        
        if task_name not in self.tasks_completed:
            self.tasks_completed.append(task_name)
    
    def get_status(self) -> Dict:
        """Get current status of the aircraft"""
        
        total_tasks = len(self.tasks)
        completed_count = len(self.tasks_completed)
        in_progress_count = len(self.tasks_in_progress)
        remaining_count = total_tasks - completed_count - in_progress_count
        
        completion_percent = (completed_count / total_tasks * 100) if total_tasks > 0 else 0
        
        return {
            'flight_id': self.flight_id,
            'aircraft_type': self.aircraft_type,
            'position': self.position,
            'total_tasks': total_tasks,
            'completed': completed_count,
            'in_progress': in_progress_count,
            'remaining': remaining_count,
            'completion_percent': completion_percent,
            'priority_score': self.priority_score,
            'current_delay': self.current_delay
        }
    
    def to_message(self) -> str:
        """Generate natural language status message"""
        
        status = self.get_status()
        
        message = f"""
Aircraft {self.flight_id} ({self.aircraft_type}) at {self.position}:
- Tasks: {status['completed']}/{status['total_tasks']} completed ({status['completion_percent']:.1f}%)
- In progress: {status['in_progress']} tasks
- Priority: {self.priority_score:.2f}
- Current delay: {self.current_delay:.1f} minutes
"""
        
        return message.strip()