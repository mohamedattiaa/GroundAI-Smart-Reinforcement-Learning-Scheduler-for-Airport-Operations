"""
Visualization utilities for the dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


class DatasetVisualizer:
    """Create visualizations of the generated dataset"""
    
    def __init__(self, flights_df: pd.DataFrame, tasks_df: pd.DataFrame):
        self.flights = flights_df
        self.tasks = tasks_df
        
        # Convert date columns
        if 'actual_arrival' in self.flights.columns:
            self.flights['actual_arrival'] = pd.to_datetime(self.flights['actual_arrival'])
        if 'arrival_time' in self.flights.columns:
            self.flights['arrival_time'] = pd.to_datetime(self.flights['arrival_time'])
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_daily_traffic(self, save_path: str = None):
        """Plot daily flight counts"""
        daily_counts = self.flights.groupby('date').size()
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        daily_counts.plot(kind='bar', ax=ax, color='steelblue', alpha=0.7)
        
        # Add trend line
        x = np.arange(len(daily_counts))
        z = np.polyfit(x, daily_counts.values, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", alpha=0.8, linewidth=2, label='Trend')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Number of Flights', fontsize=12)
        ax.set_title('Daily Flight Traffic', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_hourly_distribution(self, save_path: str = None):
        """Plot hourly arrival distribution"""
        self.flights['hour'] = self.flights['actual_arrival'].dt.hour
        hourly_counts = self.flights['hour'].value_counts().sort_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.bar(hourly_counts.index, hourly_counts.values, 
               color='darkgreen', alpha=0.7, edgecolor='black')
        
        # Highlight peak hours
        peak_hours = [7, 8, 13, 18, 19, 20]
        for hour in peak_hours:
            if hour in hourly_counts.index:
                ax.bar(hour, hourly_counts[hour], color='orange', alpha=0.9)
        
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Number of Arrivals', fontsize=12)
        ax.set_title('Hourly Arrival Distribution', fontsize=14, fontweight='bold')
        ax.set_xticks(range(6, 24))
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_delay_distribution(self, save_path: str = None):
        """Plot arrival delay distribution"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        axes[0].hist(self.flights['arrival_delay_minutes'], bins=50, 
                     color='coral', alpha=0.7, edgecolor='black')
        axes[0].axvline(self.flights['arrival_delay_minutes'].mean(), 
                       color='red', linestyle='--', linewidth=2, 
                       label=f'Mean: {self.flights["arrival_delay_minutes"].mean():.1f} min')
        axes[0].axvline(self.flights['arrival_delay_minutes'].median(), 
                       color='blue', linestyle='--', linewidth=2,
                       label=f'Median: {self.flights["arrival_delay_minutes"].median():.1f} min')
        axes[0].set_xlabel('Delay (minutes)', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Arrival Delay Distribution', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot by aircraft type
        aircraft_order = self.flights['aircraft_type'].value_counts().index
        sns.boxplot(data=self.flights, y='aircraft_type', x='arrival_delay_minutes',
                   order=aircraft_order, ax=axes[1], palette='Set2')
        axes[1].set_xlabel('Delay (minutes)', fontsize=12)
        axes[1].set_ylabel('Aircraft Type', fontsize=12)
        axes[1].set_title('Delay by Aircraft Type', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_aircraft_mix(self, save_path: str = None):
        """Plot aircraft type distribution"""
        aircraft_counts = self.flights['aircraft_type'].value_counts()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart
        colors = sns.color_palette('pastel')
        axes[0].pie(aircraft_counts.values, labels=aircraft_counts.index, 
                   autopct='%1.1f%%', startangle=90, colors=colors)
        axes[0].set_title('Aircraft Type Distribution', fontsize=14, fontweight='bold')
        
        # Bar chart
        aircraft_counts.plot(kind='bar', ax=axes[1], color=colors, edgecolor='black')
        axes[1].set_xlabel('Aircraft Type', fontsize=12)
        axes[1].set_ylabel('Number of Flights', fontsize=12)
        axes[1].set_title('Flights per Aircraft Type', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_task_durations(self, save_path: str = None):
        """Plot task duration distributions"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        task_order = self.tasks.groupby('task_name')['duration'].median().sort_values(ascending=False).index
        
        sns.violinplot(data=self.tasks, y='task_name', x='duration',
                      order=task_order, ax=ax, palette='muted')
        
        ax.set_xlabel('Duration (minutes)', fontsize=12)
        ax.set_ylabel('Task Type', fontsize=12)
        ax.set_title('Task Duration Distributions', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_position_utilization(self, save_path: str = None):
        """Plot parking position utilization"""
        position_counts = self.flights['position'].value_counts().head(20)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        position_counts.plot(kind='barh', ax=ax, color='teal', alpha=0.7)
        
        ax.set_xlabel('Number of Uses', fontsize=12)
        ax.set_ylabel('Position', fontsize=12)
        ax.set_title('Top 20 Most Used Positions', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_all_plots(self, output_dir: str = "data/statistics/visualizations"):
        """Generate all visualizations"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print("📊 Generating visualizations...\n")
        
        plots = [
            ("daily_traffic.png", self.plot_daily_traffic),
            ("hourly_distribution.png", self.plot_hourly_distribution),
            ("delay_distribution.png", self.plot_delay_distribution),
            ("aircraft_mix.png", self.plot_aircraft_mix),
            ("task_durations.png", self.plot_task_durations),
            ("position_utilization.png", self.plot_position_utilization),
        ]
        
        for filename, plot_func in plots:
            save_path = output_path / filename
            plot_func(save_path=str(save_path))
        
        print(f"\n✅ All visualizations saved to {output_path}")


def create_visualizations(
    flights_path: str = "data/raw/flight_schedules.csv",
    tasks_path: str = "data/raw/tasks.csv",
    output_dir: str = "data/statistics/visualizations"
):
    """Main function to create all visualizations"""
    print("📂 Loading dataset...")
    flights_df = pd.read_csv(flights_path)
    tasks_df = pd.read_csv(tasks_path)
    
    visualizer = DatasetVisualizer(flights_df, tasks_df)
    visualizer.create_all_plots(output_dir)


if __name__ == "__main__":
    create_visualizations()