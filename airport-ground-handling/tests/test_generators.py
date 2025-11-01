"""
Unit tests for data generators
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.data_generation.airport_config import AirportConfig
from src.data_generation.aircraft_generator import AircraftGenerator
from src.data_generation.vehicle_generator import VehicleFleet


class TestAirportConfig:
    def test_initialization(self):
        airport = AirportConfig("configs/airport_config.yaml")
        assert airport.name is not None
        assert len(airport.terminals) > 0
        assert len(airport.all_positions) > 0
    
    def test_travel_times(self):
        airport = AirportConfig("configs/airport_config.yaml")
        # Diagonal should be zero
        n = len(airport.all_positions)
        assert np.allclose(np.diag(airport.travel_time_matrix), 0)
        # Matrix should be symmetric
        assert np.allclose(airport.travel_time_matrix, 
                          airport.travel_time_matrix.T)
    
    def test_compatible_positions(self):
        airport = AirportConfig("configs/airport_config.yaml")
        positions_a320 = airport.get_compatible_positions("A320")
        positions_b777 = airport.get_compatible_positions("B777")
        
        assert len(positions_a320) > 0
        assert len(positions_b777) > 0


class TestAircraftGenerator:
    def test_generate_schedule(self):
        airport = AirportConfig("configs/airport_config.yaml")
        gen = AircraftGenerator()
        
        schedule = gen.generate_daily_schedule(
            date=datetime(2024, 1, 1),
            airport_config=airport,
            num_flights=50
        )
        
        assert len(schedule) == 50
        assert 'flight_id' in schedule.columns
        assert 'aircraft_type' in schedule.columns
        assert 'arrival_time' in schedule.columns
    
    def test_task_generation(self):
        gen = AircraftGenerator()
        
        # Create mock flight
        flight = pd.Series({
            'flight_id': 'FL0001',
            'aircraft_type': 'A320',
            'equipment_failure': False,
            'failed_task': None
        })
        
        tasks = gen.generate_task_list(flight)
        
        assert len(tasks) > 0
        assert all('task_name' in task for task in tasks)
        assert all('duration' in task for task in tasks)


class TestVehicleFleet:
    def test_initialization(self):
        fleet = VehicleFleet()
        vehicles = fleet.get_all_vehicles()
        
        assert len(vehicles) > 0
        assert all(hasattr(v, 'vehicle_id') for v in vehicles)
    
    def test_compatible_vehicles(self):
        fleet = VehicleFleet()
        
        compatible = fleet.get_compatible_vehicles(
            task_name="refueling",
            required_vehicle_types=["fuel_truck"],
            aircraft_type="A320"
        )
        
        assert len(compatible) > 0
        assert all(v.vehicle_type == "fuel_truck" for v in compatible)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])