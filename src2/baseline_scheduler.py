"""
Naive Sequential Scheduler (First Algorithm from the PDF)
--------------------------------------------------------
- Processes aircraft strictly in order of arrival.
- Executes all tasks sequentially for each aircraft (no parallelism).
- Uses the first compatible vehicle available (ignores travel/availability).
- Vehicles "teleport" to each aircraft (no travel time considered).
- Produces large cumulative delays for later aircraft (like in the PDF).
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path("data")

def to_dt(t: str) -> datetime:
    return datetime.strptime(t, "%H:%M")

def to_str(dt: datetime) -> str:
    return dt.strftime("%H:%M")

def parse_list(x):
    if pd.isna(x) or str(x).strip() == "":
        return []
    s = str(x).strip()
    if s.startswith("[") and s.endswith("]"):
        return [int(i) for i in s[1:-1].split(",") if i.strip()]
    if ";" in s:
        return [int(i) for i in s.split(";") if i.strip()]
    if "," in s:
        return [int(i) for i in s.split(",") if i.strip()]
    try:
        return [int(s)]
    except:
        return []

# === Load data ===
aircraft_df = pd.read_csv(DATA_DIR / "aircraft.csv")
tasks_df = pd.read_csv(DATA_DIR / "tasks.csv")
vehicles_df = pd.read_csv(DATA_DIR / "vehicles.csv")

vehicles_df["compatible_tasks"] = vehicles_df["compatible_tasks"].apply(parse_list)
tasks_df["precedence"] = tasks_df["precedence"].apply(parse_list)
tasks_df["task_id"] = tasks_df["task_id"].astype(int)
tasks_df = tasks_df.sort_values("task_id").reset_index(drop=True)

# === Initialize ===
vehicles = {v["vehicle_id"]: to_dt("08:00") for _, v in vehicles_df.iterrows()}  # vehicle availability
assignments = []

# === Naive sequential scheduling ===
for _, ac in aircraft_df.sort_values("start_time").iterrows():
    ac_id = ac["aircraft_id"]
    ac_arrival = to_dt(ac["start_time"])
    ac_departure_planned = to_dt(ac["end_time_planned"])
    current_time = ac_arrival

    for _, t in tasks_df.iterrows():
        tid = int(t["task_id"])
        tname = t["task_name"]
        duration = int(t["duration_min"])

        # choose first compatible vehicle
        compat = vehicles_df[vehicles_df["compatible_tasks"].apply(lambda x: tid in x)]
        if compat.empty:
            continue
        vehicle_id = compat.iloc[0]["vehicle_id"]

        # task starts after current_time (no vehicle coordination)
        start_time = current_time
        end_time = start_time + timedelta(minutes=duration)

        # update vehicle availability (but ignored in assignment)
        vehicles[vehicle_id] = end_time

        assignments.append({
            "aircraft_id": ac_id,
            "task_id": tid,
            "task_name": tname,
            "vehicle_id": vehicle_id,
            "start_time": to_str(start_time),
            "end_time": to_str(end_time),
            "delay_min": 0
        })

        # next task starts immediately after previous
        current_time = end_time

    # total delay = difference between actual last end and planned departure
    total_delay = (current_time - ac_departure_planned).total_seconds() / 60.0
    if total_delay < 0:
        total_delay = 0

    print(f"Aircraft {ac_id} scheduled. Total delay: {round(total_delay,2)} min")

# === Save results ===
df = pd.DataFrame(assignments)
df.to_csv("naive_schedule.csv", index=False)
print("\n✅ Naive sequential scheduling done. Saved: naive_schedule.csv")
