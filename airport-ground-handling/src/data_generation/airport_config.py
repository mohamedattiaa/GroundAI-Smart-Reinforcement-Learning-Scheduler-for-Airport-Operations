"""
Airport configuration and layout management
"""

import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, field


@dataclass
class Terminal:
    """Represents an airport terminal"""
    name: str
    num_gates: int
    aircraft_types: List[str]
    center_coordinates: Tuple[float, float]
    gates: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate gate positions around terminal center"""
        if not self.gates:
            self.gates = self._generate_gate_positions()
    
    def _generate_gate_positions(self) -> Dict[str, Tuple[float, float]]:
        """Generate gate coordinates in a semicircle around terminal"""
        gates = {}
        cx, cy = self.center_coordinates
        radius = 30  # meters
        
        for i in range(self.num_gates):
            angle = np.pi * i / (self.num_gates - 1)  # Semicircle
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            gates[f"{self.name}_gate_{i+1}"] = (x, y)
        
        return gates


class AirportConfig:
    """
    Manages airport layout, terminals, and operational parameters
    """
    
    def __init__(self, config_path: str = "configs/airport_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        self.name = self.config['airport']['name']
        self.code = self.config['airport']['code']
        
        # Initialize terminals
        self.terminals = self._create_terminals()
        
        # Initialize remote stands
        self.remote_stands = self._create_remote_stands()
        
        # Service base
        self.base_location = tuple(
            self.config['airport']['layout']['service_base']['coordinates'].values()
        )
        
        # All positions
        self.all_positions = self._collect_all_positions()
        
        # Travel time matrix
        self.travel_time_matrix = self._calculate_travel_times()
    
    def _load_config(self) -> dict:
        """Load YAML configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {self.config_path.absolute()}\n"
                f"Please create the file with proper YAML syntax."
            )
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(
                f"Config file is empty or invalid: {self.config_path}\n"
                f"Please check YAML syntax."
            )
        
        if 'airport' not in config:
            raise ValueError(
                f"Config file missing 'airport' key: {self.config_path}\n"
                f"Top-level keys found: {list(config.keys())}"
            )
        
        return config
    
    def _create_terminals(self) -> Dict[str, Terminal]:
        """Create terminal objects from config"""
        terminals = {}
        
        for term_name, term_config in self.config['airport']['layout']['terminals'].items():
            terminal = Terminal(
                name=term_name,
                num_gates=term_config['gates'],
                aircraft_types=term_config['type'],
                center_coordinates=(
                    term_config['coordinates']['x'],
                    term_config['coordinates']['y']
                )
            )
            terminals[term_name] = terminal
        
        return terminals
    
    def _create_remote_stands(self) -> Dict[str, Tuple[float, float]]:
        """Generate remote stand positions"""
        stands = {}
        config = self.config['airport']['layout']['remote_stands']
        
        bounds = config['area_bounds']
        num_stands = config['count']
        
        # Random positions within bounds
        for i in range(num_stands):
            x = np.random.uniform(bounds['x_min'], bounds['x_max'])
            y = np.random.uniform(bounds['y_min'], bounds['y_max'])
            stands[f"remote_stand_{i+1}"] = (x, y)
        
        return stands
    
    def _collect_all_positions(self) -> Dict[str, Tuple[float, float]]:
        """Collect all parking positions"""
        positions = {'base': self.base_location}
        
        # Add all gates from terminals
        for terminal in self.terminals.values():
            positions.update(terminal.gates)
        
        # Add remote stands
        positions.update(self.remote_stands)
        
        return positions
    
    def _calculate_travel_times(self) -> np.ndarray:
        """
        Calculate travel time matrix between all positions
        Returns: NxN matrix where N = number of positions
        """
        positions_list = list(self.all_positions.keys())
        n = len(positions_list)
        
        travel_times = np.zeros((n, n))
        avg_speed_kmh = 20  # Average apron speed
        
        for i, pos1 in enumerate(positions_list):
            for j, pos2 in enumerate(positions_list):
                if i == j:
                    travel_times[i][j] = 0
                else:
                    # Euclidean distance
                    x1, y1 = self.all_positions[pos1]
                    x2, y2 = self.all_positions[pos2]
                    distance_m = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    
                    # Convert to time (minutes)
                    time_hours = distance_m / 1000 / avg_speed_kmh
                    travel_times[i][j] = time_hours * 60
        
        return travel_times
    
    def get_compatible_positions(self, aircraft_type: str) -> List[str]:
        """Get list of positions compatible with aircraft type"""
        compatible = []
        
        # Determine if narrow-body or wide-body
        narrow_body_types = ['A320', 'B737', 'A321']
        wide_body_types = ['B777', 'A350', 'B787', 'A330']
        
        is_narrow_body = aircraft_type in narrow_body_types
        is_wide_body = aircraft_type in wide_body_types
        
        # Check terminals
        for term_name, terminal in self.terminals.items():
            # Check if terminal accepts this aircraft
            terminal_types = terminal.aircraft_types
            
            # Terminal accepts if:
            # 1. 'all' is in list
            # 2. Specific aircraft type is listed
            # 3. 'narrow_body' is listed and aircraft is narrow-body
            # 4. 'wide_body' is listed and aircraft is wide-body
            
            accepts = False
            
            if 'all' in terminal_types:
                accepts = True
            elif aircraft_type in terminal_types:
                accepts = True
            elif 'narrow_body' in terminal_types and is_narrow_body:
                accepts = True
            elif 'wide_body' in terminal_types and is_wide_body:
                accepts = True
            
            if accepts:
                compatible.extend(terminal.gates.keys())
        
        # Remote stands (usually narrow-body only)
        if is_narrow_body:
            compatible.extend(self.remote_stands.keys())
        
        # Safety check: if no compatible positions found, use all positions except base
        if len(compatible) == 0:
            print(f"⚠️  Warning: No compatible positions for {aircraft_type}, using all gates")
            for terminal in self.terminals.values():
                compatible.extend(terminal.gates.keys())
            # Also add remote stands as fallback
            compatible.extend(self.remote_stands.keys())
        
        return compatible

    def save_layout(self, output_path: str):
        """Save airport layout as JSON"""
        import json
        
        layout_data = {
            'airport_name': self.name,
            'airport_code': self.code,
            'base_location': self.base_location,
            'terminals': {
                name: {
                    'gates': list(term.gates.keys()),
                    'coordinates': term.center_coordinates
                }
                for name, term in self.terminals.items()
            },
            'remote_stands': self.remote_stands,
            'all_positions': self.all_positions
        }
        
        with open(output_path, 'w') as f:
            json.dump(layout_data, f, indent=2)
        
        print(f"✅ Airport layout saved to {output_path}")