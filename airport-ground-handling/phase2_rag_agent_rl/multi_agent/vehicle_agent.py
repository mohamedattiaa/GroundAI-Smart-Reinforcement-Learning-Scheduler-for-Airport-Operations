"""
Vehicle Agent - Represents a ground service vehicle
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class VehicleState:
    """State information for a vehicle"""
    vehicle_id: str
    vehicle_type: str
    current_position: str
    available_at: pd.Timestamp
    tasks_completed: int
    max_tasks: int
    compatible_aircraft: List[str]


class VehicleAgent:
    """
    Agent representing a ground service vehicle
    Responsible for managing schedule and responding to requests
    """
    
    def __init__(
        self,
        vehicle_id: str,
        vehicle_type: str,
        max_tasks: int,
        compatible_aircraft: List[str],
        base_position: str = "base"
    ):
        self.vehicle_id = vehicle_id
        self.vehicle_type = vehicle_type
        self.max_tasks = max_tasks
        self.compatible_aircraft = compatible_aircraft
        self.base_position = base_position
        
        # State
        self.current_position = base_position
        self.available_at = pd.Timestamp.now()
        self.tasks_completed = 0
        self.schedule = []  # List of assigned tasks
        
        # Performance metrics
        self.total_distance = 0.0
        self.total_idle_time = 0.0
    
    def can_serve(self, request: Dict) -> bool:
        """Check if this vehicle can serve a request"""
        
        # Check vehicle type compatibility
        if self.vehicle_type not in request['required_vehicles']:
            return False
        
        # Check aircraft compatibility
        aircraft_type = request['aircraft_type']
        if 'all' not in self.compatible_aircraft:
            if aircraft_type not in self.compatible_aircraft:
                return False
        
        # Check if need to return to base
        if self.tasks_completed >= self.max_tasks:
            return False
        
        return True
    
    def calculate_availability(
        self,
        request: Dict,
        travel_time: float
    ) -> pd.Timestamp:
        """Calculate when vehicle would be available for this request"""
        
        # Travel time from current position to request location
        arrival_time = self.available_at + pd.Timedelta(minutes=travel_time)
        
        return arrival_time
    
    def bid_on_request(
        self,
        request: Dict,
        travel_time: float
    ) -> Optional[Dict]:
        """Submit a bid to serve a request"""
        
        if not self.can_serve(request):
            return None
        
        availability = self.calculate_availability(request, travel_time)
        
        # Calculate bid score (lower is better)
        # Factors: availability time, current workload, distance
        
        time_score = (availability - pd.Timestamp.now()).total_seconds() / 60
        workload_score = self.tasks_completed / self.max_tasks * 100
        distance_score = travel_time
        
        # Weighted sum
        bid_score = (
            time_score * 0.5 +
            workload_score * 0.3 +
            distance_score * 0.2
        )
        
        bid = {
            'vehicle_id': self.vehicle_id,
            'vehicle_type': self.vehicle_type,
            'available_at': availability,
            'bid_score': bid_score,
            'current_position': self.current_position,
            'tasks_completed': self.tasks_completed
        }
        
        return bid
    
    def assign_task(
        self,
        request: Dict,
        start_time: pd.Timestamp,
        duration: float
    ):
        """Assign a task to this vehicle"""
        
        task_assignment = {
            'flight_id': request['flight_id'],
            'task_name': request['task_name'],
            'position': request['position'],
            'start_time': start_time,
            'end_time': start_time + pd.Timedelta(minutes=duration),
            'duration': duration
        }
        
        self.schedule.append(task_assignment)
        self.current_position = request['position']
        self.available_at = task_assignment['end_time']
        self.tasks_completed += 1
    
    def needs_base_return(self) -> bool:
        """Check if vehicle should return to base"""
        return self.tasks_completed >= self.max_tasks
    
    def return_to_base(self, travel_time: float):
        """Return vehicle to base"""
        self.current_position = self.base_position
        self.available_at = self.available_at + pd.Timedelta(minutes=travel_time)
        self.tasks_completed = 0
    
    def get_status(self) -> Dict:
        """Get current status"""
        
        utilization = (self.tasks_completed / self.max_tasks * 100) if self.max_tasks > 0 else 0
        
        return {
            'vehicle_id': self.vehicle_id,
            'vehicle_type': self.vehicle_type,
            'current_position': self.current_position,
            'available_at': str(self.available_at),
            'tasks_completed': self.tasks_completed,
            'max_tasks': self.max_tasks,
            'utilization': utilization,
            'schedule_length': len(self.schedule)
        }
    
    def to_message(self) -> str:
        """Generate natural language status message"""
        
        status = self.get_status()
        
        message = f"""
Vehicle {self.vehicle_id} ({self.vehicle_type}):
- Position: {self.current_position}
- Available: {status['available_at']}
- Workload: {status['tasks_completed']}/{self.max_tasks} ({status['utilization']:.1f}%)
- Scheduled tasks: {status['schedule_length']}
"""
        
        return message.strip()