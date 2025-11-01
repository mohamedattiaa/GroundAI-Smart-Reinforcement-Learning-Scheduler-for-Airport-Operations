# Airport Ground Handling Synthetic Dataset Generator

A comprehensive synthetic dataset generator for airport ground service operations, designed for optimization, scheduling, and machine learning research.

## 🎯 Features

- **Realistic Flight Schedules**: Multi-day schedules with peak hour patterns
- **Aircraft Diversity**: 5 aircraft types (A320, B737, A321, B777, A350)
- **Ground Service Tasks**: Complete turnaround operations with precedence constraints
- **Vehicle Fleet Management**: Multiple vehicle types with capacity constraints
- **Stochastic Elements**: Delays, weather impacts, equipment failures
- **Scenario Chunking**: Pre-processed scenarios for RAG/RL training

## 📦 Installation
```bash
# Clone repository
git clone <your-repo-url>
cd airport-ground-handling

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### Generate Dataset
```bash
# Generate 90 days of data
python generate_dataset.py --days 90 --output data/raw

# Custom configuration
python generate_dataset.py --days 30 --chunk-size 50 --no-scenarios
```

### Validate Dataset
```python
from src.validation.constraint_checker import validate_dataset

results = validate_dataset(
    flights_path="data/raw/flight_schedules.csv",
    tasks_path="data/raw/tasks.csv"
)
```

### Create Visualizations
```python
from src.visualization.plots import create_visualizations

create_visualizations(
    flights_path="data/raw/flight_schedules.csv",
    tasks_path="data/raw/tasks.csv"
)
```

## 📁 Output Files

After generation, you'll have:
```
data/
├── raw/
│   ├── flight_schedules.csv     # All flight information
│   ├── tasks.csv                # Task details for each flight
│   ├── aircraft_specs.json      # Aircraft type specifications
│   ├── vehicle_fleet.json       # Vehicle fleet configuration
│   ├── airport_layout.json      # Airport layout and positions
│   └── travel_times.csv         # Travel time matrix
│
├── processed/
│   └── scenarios/               # 100-flight scenario chunks
│       ├── scenario_0001.json
│       ├── scenario_0002.json
│       └── ...
│
└── statistics/
    ├── dataset_summary.txt      # Statistical summary
    └── visualizations/          # All plots
        ├── daily_traffic.png
        ├── delay_distribution.png
        └── ...
```

## 📊 Dataset Statistics (Example)

- **Total Flights**: 10,800 (90 days × 120 avg/day)
- **Aircraft Mix**: A320 (35%), B737 (30%), A321 (15%), B777 (12%), A350 (8%)
- **Total Tasks**: ~97,200 (9-10 tasks per flight)
- **Average Delay**: 8.5 minutes
- **Equipment Failures**: 5% of flights

## 🎨 Customization

### Modify Aircraft Types

Edit `configs/aircraft_types.yaml`:
```yaml
B787:
  category: wide_body
  pax_capacity: 330
  typical_turnaround: 85
  tasks:
    # Define your tasks here
```

### Adjust Traffic Patterns

Edit `configs/generation_config.yaml`:
```yaml
generation:
  flights_per_day:
    weekday: {min: 150, max: 200, mean: 175}
```

### Change Airport Layout

Edit `configs/airport_config.yaml`:
```yaml
airport:
  layout:
    terminals:
      terminal_E:
        gates: 15
        type: [narrow_body, wide_body]
```

## 🧪 Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_generators.py::TestAirportConfig -v
```

## 📚 Use Cases

### 1. Optimization Research
Use the dataset to benchmark scheduling algorithms

### 2. Machine Learning
Train models to predict delays or optimize resource allocation

### 3. RAG Systems
Use scenario chunks to build retrieval-augmented generation systems

### 4. Reinforcement Learning
Use as environment for training RL agents

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## 📝 License

MIT License - see LICENSE file for details

## 📧 Contact

For questions or issues, please open a GitHub issue.

## 🙏 Acknowledgments

Based on research from:
- "Gestion des moyens au sol d'une plate-forme aéroportuaire" (ENAC, 2009-2010)
- Real-world airport operations data and best practices