import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# === Load data ===
schedule = pd.read_csv("optimized_schedule.csv")

# Convert time strings to datetime
def to_dt(t):
    return datetime.strptime(t, "%H:%M")

schedule["start_dt"] = schedule["start_time"].apply(to_dt)
schedule["end_dt"] = schedule["end_time"].apply(to_dt)

# Unique colors per vehicle
vehicles = schedule["vehicle_id"].unique()
colors = {v: plt.cm.tab20(i / len(vehicles)) for i, v in enumerate(vehicles)}

# === Plot ===
fig, ax = plt.subplots(figsize=(14, 6))
y_positions = {ac: i for i, ac in enumerate(sorted(schedule["aircraft_id"].unique()))}

for _, row in schedule.iterrows():
    y = y_positions[row["aircraft_id"]]
    start = row["start_dt"]
    end = row["end_dt"]
    color = colors[row["vehicle_id"]]
    ax.barh(
        y,
        (end - start).seconds / 60,
        left=(start - schedule["start_dt"].min()).seconds / 60,
        color=color,
        edgecolor="black",
        height=0.5,
    )

ax.set_yticks(list(y_positions.values()))
ax.set_yticklabels(list(y_positions.keys()))
ax.set_xlabel("Minutes since start of day")
ax.set_ylabel("Aircraft")
ax.set_title("🛫 Optimized Aircraft Turnaround Gantt Chart (Vehicle Colors)")
ax.grid(True, axis="x", linestyle="--", alpha=0.5)

# Legend (vehicles)
patches = [mpatches.Patch(color=colors[v], label=v) for v in vehicles]
ax.legend(handles=patches, title="Vehicles", loc="upper right")

plt.tight_layout()
plt.show()
