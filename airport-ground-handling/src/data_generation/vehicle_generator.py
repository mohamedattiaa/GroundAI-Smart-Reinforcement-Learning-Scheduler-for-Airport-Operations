"""
Vehicle fleet generation and management
"""

import yaml
import pandas as pd
import numpy as np
from typing import Dict, List
from dataclasses import dataclass


@dataclass
class Vehicle:
    """Represents a single ground service vehicle"""
    vehicle_id: str
    vehicle_type: str
    max_tasks_before_base: int
    speed_kmh: float
    compatible_aircraft: List[str]
    current_position: str = "base"
    available_at: float = 0.0  # minutes from start of day
    tasks_completed: int = 0


class VehicleFleet:
    """
    Manages the airport's ground service vehicle fleet
    """
    
    def __init__(self, config_path: str = "configs/airport_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.fleet_config = self.config['airport']['vehicle_fleet']
        self.fleet = self._initialize_fleet()  # Initialize fleet here!
    
    def _initialize_fleet(self) -> Dict[str, List[Vehicle]]:
        """Create all vehicles from configuration"""
        fleet = {}
        
        for vehicle_type, specs in self.fleet_config.items():
            fleet[vehicle_type] = []
            
            for i in range(specs['count']):
                # Handle compatible_aircraft - it might be a list or a string
                compatible = specs['compatible_aircraft']
                if isinstance(compatible, str):
                    compatible = [compatible]
                elif not isinstance(compatible, list):
                    compatible = ['all']
                
                vehicle = Vehicle(
                    vehicle_id=f"{vehicle_type}_{i+1}",
                    vehicle_type=vehicle_type,
                    max_tasks_before_base=specs['max_tasks_before_base'],
                    speed_kmh=specs['speed_kmh'],
                    compatible_aircraft=compatible
                )
                fleet[vehicle_type].append(vehicle)
        
        return fleet
    
    def get_all_vehicles(self) -> List[Vehicle]:
        """Get flat list of all vehicles"""
        all_vehicles = []
        for vehicle_list in self.fleet.values():
            all_vehicles.extend(vehicle_list)
        return all_vehicles
    
    def get_compatible_vehicles(
        self,
        task_name: str,
        required_vehicle_types: List[str],
        aircraft_type: str
    ) -> List[Vehicle]:
        """
        Find vehicles that can perform a specific task
        
        Args:
            task_name: Name of the task
            required_vehicle_types: List of vehicle types that can do this task
            aircraft_type: Type of aircraft
        
        Returns:
            List of compatible Vehicle objects
        """
        compatible = []
        
        for req_type in required_vehicle_types:
            if req_type in self.fleet:
                for vehicle in self.fleet[req_type]:
                    # Check aircraft compatibility
                    if ('all' in vehicle.compatible_aircraft or 
                        aircraft_type in vehicle.compatible_aircraft):
                        compatible.append(vehicle)
        
        return compatible
    
    def reset_fleet(self):
        """Reset all vehicles to initial state"""
        for vehicle_list in self.fleet.values():
            for vehicle in vehicle_list:
                vehicle.current_position = "base"
                vehicle.available_at = 0.0
                vehicle.tasks_completed = 0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Export fleet as DataFrame"""
        data = []
        
        for vehicle in self.get_all_vehicles():
            data.append({
                'vehicle_id': vehicle.vehicle_id,
                'vehicle_type': vehicle.vehicle_type,
                'max_tasks': vehicle.max_tasks_before_base,
                'speed_kmh': vehicle.speed_kmh,
                'compatible_aircraft': ','.join(vehicle.compatible_aircraft),
                'current_position': vehicle.current_position,
                'available_at': vehicle.available_at,
                'tasks_completed': vehicle.tasks_completed
            })
        
        return pd.DataFrame(data)
    
    def save_fleet_config(self, output_path: str):
        """Save fleet configuration to JSON"""
        import json
        
        fleet_data = {}
        for vehicle_type, vehicle_list in self.fleet.items():
            if len(vehicle_list) == 0:
                continue
                
            fleet_data[vehicle_type] = {
                'count': len(vehicle_list),
                'specs': {
                    'max_tasks_before_base': vehicle_list[0].max_tasks_before_base,
                    'speed_kmh': vehicle_list[0].speed_kmh,
                    'compatible_aircraft': vehicle_list[0].compatible_aircraft
                },
                'vehicles': [v.vehicle_id for v in vehicle_list]
            }
        
        with open(output_path, 'w') as f:
            json.dump(fleet_data, f, indent=2)
        
        print(f"✅ Vehicle fleet configuration saved to {output_path}")