# === Gantt Chart Visualization for Airport Ground Operations ===
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
# --- Load generated dataset ---
tasks = pd.read_csv("data/tasks.csv")
assignments = pd.read_csv("data/assignments.csv")

# Merge to get full info (task type + times + vehicle)
df = pd.merge(
    assignments,
    tasks[["task_id", "task_type", "required_vehicle_type", "aircraft_id"]],
    on="task_id",
    how="left",
    suffixes=("", "_task")
)

# If aircraft_id doesn't exist, fix from merged columns
if "aircraft_id" not in df.columns:
    if "aircraft_id_task" in df.columns:
        df["aircraft_id"] = df["aircraft_id_task"]
# Convert times
def to_minutes(t):
    return int(datetime.strptime(t, "%H:%M").hour)*60 + int(datetime.strptime(t, "%H:%M").minute)
df["start_min"] = df["start_time"].apply(to_minutes)
df["end_min"] = df["end_time"].apply(to_minutes)

# Sort by aircraft then start time
df = df.sort_values(by=["aircraft_id","start_min"])

# --- Define colors per task type ---
colors = {
    "Deboarding": "#6baed6",
    "Refueling": "#fd8d3c",
    "Cleaning": "#74c476",
    "Boarding": "#9e9ac8",
    "Catering": "#fdae6b"
}

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6))

y_labels = sorted(df["aircraft_id"].unique())
y_pos = {ac:i for i,ac in enumerate(y_labels)}

for _, row in df.iterrows():
    start = row["start_min"]
    end = row["end_min"]
    duration = end - start
    color = colors.get(row["task_type"], "#cccccc")
    y = y_pos[row["aircraft_id"]]
    ax.barh(y, duration, left=start, height=0.4, color=color, edgecolor='black')
    ax.text(start + duration/2, y, f"{row['task_type']} ({row['vehicle_id']})", 
            ha="center", va="center", fontsize=8, color="black")

ax.set_yticks(list(y_pos.values()))
ax.set_yticklabels(y_labels)
ax.set_xlabel("Time (minutes since midnight)")
ax.set_title("Aircraft Ground Handling Schedule (Gantt Chart)")

# Create legend
patches = [mpatches.Patch(color=v, label=k) for k,v in colors.items()]
ax.legend(handles=patches, loc="upper right")

plt.tight_layout()
plt.show()
