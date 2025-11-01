#!/usr/bin/env python3
"""
Main script to generate airport ground handling synthetic dataset

Usage:
    python generate_dataset.py --days 90 --output data/raw
"""

import argparse
from pathlib import Path
from src.data_generation.scenario_generator import ScenarioGenerator


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic airport ground handling dataset"
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Number of days to generate (default: 90)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory for generated files (default: data/raw)'
    )
    
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=100,
        help='Number of flights per scenario chunk (default: 100)'
    )
    
    parser.add_argument(
        '--no-scenarios',
        action='store_true',
        help='Skip scenario chunk generation'
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("AIRPORT GROUND HANDLING SYNTHETIC DATASET GENERATOR")
    print("=" * 70)
    
    # Initialize generator
    generator = ScenarioGenerator()
    
    # Generate main dataset
    output_files = generator.generate_full_dataset(
        output_dir=args.output,
        num_days=args.days
    )
    
    # Generate scenario chunks
    if not args.no_scenarios:
        import pandas as pd
        
        flights_df = pd.read_csv(output_files['flights'])
        tasks_df = pd.read_csv(output_files['tasks'])
        
        generator.generate_scenario_chunks(
            flights_df=flights_df,
            tasks_df=tasks_df,
            output_dir='data/processed/scenarios',
            chunk_size=args.chunk_size
        )
    
    print("\n" + "=" * 70)
    print("✅ DATASET GENERATION COMPLETE!")
    print("=" * 70)
    print("\n📂 Generated files:")
    for key, path in output_files.items():
        print(f"   - {key}: {path}")
    
    print("\n📊 Next steps:")
    print("   1. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
    print("   2. Validate: python -m pytest tests/")
    print("   3. Visualize: python src/visualization/plots.py")
    print("\n")


if __name__ == "__main__":
    main()