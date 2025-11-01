import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

# === Setup ===
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

# Parameters
NUM_AIRCRAFT = 12
NUM_TASKS = random.randint(4, 6)
NUM_VEHICLES = random.randint(4, 5)
START_DAY = datetime.strptime("08:00", "%H:%M")
MIN_GAP = timedelta(minutes=15)
POSITIONS = [1, 2, 3, 4]

# === Helper Functions ===
def format_time(dt): return dt.strftime("%H:%M")

def next_available_time(parking_schedule, position):
    if position not in parking_schedule or not parking_schedule[position]:
        return START_DAY
    last_end = max([e for (_, e) in parking_schedule[position]])
    return last_end + MIN_GAP

# === Initialize schedule tracking ===
parking_schedule = {pos: [] for pos in POSITIONS}
aircraft_data = []

for i in range(NUM_AIRCRAFT):
    duration = timedelta(minutes=random.randint(25, 50))
    random.shuffle(POSITIONS)
    for pos in POSITIONS:
        available_start = next_available_time(parking_schedule, pos)
        start = available_start
        end = start + duration
        parking_schedule[pos].append((start, end))
        aircraft_data.append([
            f"A{i+1}", pos, format_time(start), format_time(end)
        ])
        break

aircraft_df = pd.DataFrame(aircraft_data, columns=["aircraft_id", "parking_position", "start_time", "end_time_planned"])

# === Tasks ===
all_tasks = ["Deboarding", "Refueling", "Catering", "Cleaning", "Boarding", "Baggage Loading"]
task_ids = list(range(1, NUM_TASKS + 1))
chosen_tasks = random.sample(all_tasks, NUM_TASKS)

durations = np.random.randint(5, 15, size=NUM_TASKS)
precedence = {1:[2,3], 2:[4], 3:[4,5], 4:[6], 5:[], 6:[]}  # simplified precedence

task_df = pd.DataFrame({
    "task_id": task_ids,
    "task_name": chosen_tasks,
    "duration_min": durations,
    "precedence": [precedence[i] if i in precedence else [] for i in task_ids]
})

# === Vehicles ===
vehicles = []
for i in range(NUM_VEHICLES):
    compatible = random.sample(task_ids, random.randint(2, NUM_TASKS))
    vehicles.append([f"V{i+1}", compatible])
vehicle_df = pd.DataFrame(vehicles, columns=["vehicle_id", "compatible_tasks"])

# === Travel Times ===
matrix = np.random.randint(2, 6, size=(NUM_VEHICLES+1, NUM_VEHICLES+1))
np.fill_diagonal(matrix, 0)
travel_df = pd.DataFrame(matrix, columns=[f"P{i}" for i in range(NUM_VEHICLES+1)], index=[f"P{i}" for i in range(NUM_VEHICLES+1)])

# === Save files ===
aircraft_df.to_csv(data_dir / "aircraft.csv", index=False)
task_df.to_csv(data_dir / "tasks.csv", index=False)
vehicle_df.to_csv(data_dir / "vehicles.csv", index=False)
travel_df.to_csv(data_dir / "travel_time.csv")

print("✅ Larger dataset generated successfully in /data!")
print(aircraft_df.head())
