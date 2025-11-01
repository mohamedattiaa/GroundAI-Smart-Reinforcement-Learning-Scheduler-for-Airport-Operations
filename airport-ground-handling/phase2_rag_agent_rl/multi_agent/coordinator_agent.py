"""
Coordinator Agent - Manages overall operations and resolves conflicts
"""

from typing import List, Dict, Optional
import pandas as pd
from .aircraft_agent import AircraftAgent
from .vehicle_agent import VehicleAgent


class CoordinatorAgent:
    """
    Central coordinator that manages aircraft and vehicle agents
    Responsible for matching requests with resources
    """
    
    def __init__(
        self,
        aircraft_agents: List[AircraftAgent],
        vehicle_agents: List[VehicleAgent],
        travel_times: pd.DataFrame
    ):
        self.aircraft_agents = {a.flight_id: a for a in aircraft_agents}
        self.vehicle_agents = {v.vehicle_id: v for v in vehicle_agents}
        self.travel_times = travel_times
        
        # Tracking
        self.assignments = []
        self.conflicts = []
        self.total_delay = 0.0
    
    def get_travel_time(
        self,
        from_position: str,
        to_position: str
    ) -> float:
        """Get travel time between positions"""
        
        try:
            return self.travel_times.loc[from_position, to_position]
        except:
            # Default if not found
            return 5.0
    
    def collect_requests(self) -> List[Dict]:
        """Collect all pending service requests from aircraft"""
        
        requests = []
        
        for aircraft in self.aircraft_agents.values():
            ready_tasks = aircraft.get_next_tasks()
            
            for task in ready_tasks:
                request = aircraft.request_service(task)
                requests.append(request)
        
        return requests
    
    def collect_bids(
        self,
        request: Dict
    ) -> List[Dict]:
        """Collect bids from vehicles for a request"""
        
        bids = []
        
        for vehicle in self.vehicle_agents.values():
            # Calculate travel time
            travel_time = self.get_travel_time(
                vehicle.current_position,
                request['position']
            )
            
            # Get bid
            bid = vehicle.bid_on_request(request, travel_time)
            
            if bid is not None:
                bids.append(bid)
        
        return bids
    
    def select_best_bid(
        self,
        bids: List[Dict],
        request: Dict
    ) -> Optional[Dict]:
        """Select the best bid for a request"""
        
        if not bids:
            return None
        
        # Sort by bid score (lower is better)
        bids.sort(key=lambda b: b['bid_score'])
        
        return bids[0]
    
    def assign_request(
        self,
        request: Dict,
        bid: Dict
    ) -> bool:
        """Assign a request to a vehicle"""
        
        vehicle_id = bid['vehicle_id']
        vehicle = self.vehicle_agents[vehicle_id]
        aircraft_id = request['flight_id']
        aircraft = self.aircraft_agents[aircraft_id]
        
        # Assign task
        vehicle.assign_task(
            request,
            bid['available_at'],
            request['duration']
        )
        
        # Update aircraft
        aircraft.assign_vehicle(request['task_name'], vehicle_id)
        
        # Record assignment
        assignment = {
            'flight_id': aircraft_id,
            'task_name': request['task_name'],
            'vehicle_id': vehicle_id,
            'start_time': bid['available_at'],
            'duration': request['duration']
        }
        
        self.assignments.append(assignment)
        
        return True
    
    def run_scheduling_round(self) -> Dict:
        """Run one round of scheduling"""
        
        # Collect all requests
        requests = self.collect_requests()
        
        if not requests:
            return {'requests': 0, 'assignments': 0, 'conflicts': 0}
        
        # Sort requests by priority and urgency
        requests.sort(
            key=lambda r: (r['priority'], r['urgency']),
            reverse=True
        )
        
        assignments_made = 0
        conflicts = 0
        
        for request in requests:
            # Collect bids
            bids = self.collect_bids(request)
            
            if not bids:
                conflicts += 1
                self.conflicts.append({
                    'request': request,
                    'reason': 'No available vehicles'
                })
                continue
            
            # Select best bid
            best_bid = self.select_best_bid(bids, request)
            
            # Assign
            if self.assign_request(request, best_bid):
                assignments_made += 1
        
        return {
            'requests': len(requests),
            'assignments': assignments_made,
            'conflicts': conflicts
        }
    
    def get_overall_status(self) -> Dict:
        """Get status of all operations"""
        
        # Aircraft status
        aircraft_status = {
            'total': len(self.aircraft_agents),
            'completed': sum(
                1 for a in self.aircraft_agents.values()
                if a.get_status()['completion_percent'] == 100
            ),
            'in_progress': len(self.aircraft_agents) - sum(
                1 for a in self.aircraft_agents.values()
                if a.get_status()['completion_percent'] == 100
            )
        }
        
        # Vehicle status
        vehicle_status = {
            'total': len(self.vehicle_agents),
            'busy': sum(
                1 for v in self.vehicle_agents.values()
                if v.available_at > pd.Timestamp.now()
            ),
            'available': sum(
                1 for v in self.vehicle_agents.values()
                if v.available_at <= pd.Timestamp.now()
            )
        }
        
        return {
            'aircraft': aircraft_status,
            'vehicles': vehicle_status,
            'assignments': len(self.assignments),
            'conflicts': len(self.conflicts)
        }
    
    def generate_report(self) -> str:
        """Generate operations report"""
        
        status = self.get_overall_status()
        
        report = f"""
AIRPORT OPERATIONS STATUS
{'='*60}

AIRCRAFT:
- Total: {status['aircraft']['total']}
- Completed: {status['aircraft']['completed']}
- In Progress: {status['aircraft']['in_progress']}

VEHICLES:
- Total: {status['vehicles']['total']}
- Busy: {status['vehicles']['busy']}
- Available: {status['vehicles']['available']}

OPERATIONS:
- Total Assignments: {status['assignments']}
- Conflicts: {status['conflicts']}

"""
        
        return report