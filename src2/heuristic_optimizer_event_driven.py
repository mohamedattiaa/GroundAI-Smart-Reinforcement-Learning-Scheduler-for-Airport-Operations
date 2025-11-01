import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

DATA_DIR = Path("data")

# === Load data ===
aircraft_df = pd.read_csv(DATA_DIR / "aircraft.csv")
tasks_df = pd.read_csv(DATA_DIR / "tasks.csv")
vehicles_df = pd.read_csv(DATA_DIR / "vehicles.csv")
travel_df = pd.read_csv(DATA_DIR / "travel_time.csv", index_col=0)

travel_df.index = travel_df.index.map(str)
travel_df.columns = travel_df.columns.map(str)
vehicles_df["compatible_tasks"] = vehicles_df["compatible_tasks"].apply(eval)

def parse_time(t): return datetime.strptime(t, "%H:%M")
for col in ["start_time", "end_time_planned"]:
    if col in aircraft_df.columns:
        aircraft_df[col] = aircraft_df[col].apply(parse_time)

def get_travel_time(p1, p2):
    if p1 == p2:
        return 0
    try:
        return int(travel_df.loc[str(p1), str(p2)])
    except Exception:
        return 3 * abs(int(str(p1).strip("P")) - int(str(p2).strip("P")))

# === Build dictionaries ===
aircraft_info = {
    row["aircraft_id"]: {
        "arrival": row["start_time"],
        "planned_end": row["end_time_planned"],
        "parking": f"P{row['parking_position']}",
    }
    for _, row in aircraft_df.iterrows()
}

vehicles = {
    v["vehicle_id"]: {
        "position": "P1",
        "available_at": aircraft_df["start_time"].min(),
        "compatible": v["compatible_tasks"],
        "assigned": [],
    }
    for _, v in vehicles_df.iterrows()
}

task_meta = {
    t["task_id"]: {"dur": t["duration_min"], "name": t["task_name"]}
    for _, t in tasks_df.iterrows()
}

# === Main heuristic loop ===
assignments = []

for ac_id, ac in sorted(aircraft_info.items(), key=lambda x: x[1]["arrival"]):
    arrival = ac["arrival"]
    parking = ac["parking"]
    planned_end = ac["planned_end"]

    # Each aircraft has all tasks initially "ready"
    ready_tasks = list(task_meta.keys())
    current_time = arrival

    while ready_tasks:
        best_vehicle, best_task = None, None
        best_metric = None
        best_start, best_end = None, None

        for tid in ready_tasks:
            for vid, v in vehicles.items():
                if tid not in v["compatible"]:
                    continue

                travel = get_travel_time(v["position"], parking)
                available = v["available_at"] + timedelta(minutes=travel)
                start_time = max(current_time, available)
                end_time = start_time + timedelta(minutes=task_meta[tid]["dur"])

                projected_delay = max(0, (end_time - planned_end).total_seconds() / 60)
                metric = (projected_delay, start_time, end_time)

                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    best_vehicle = vid
                    best_task = tid
                    best_start = start_time
                    best_end = end_time

        if best_vehicle is None or best_task is None:
            break  # if no compatible vehicle available

        # Assign the best task to the best vehicle
        v = vehicles[best_vehicle]
        v["available_at"] = best_end
        v["position"] = parking
        v["assigned"].append((ac_id, best_task))

        assignments.append({
            "aircraft_id": ac_id,
            "task_id": best_task,
            "task_name": task_meta[best_task]["name"],
            "vehicle_id": best_vehicle,
            "start_time": best_start.strftime("%H:%M"),
            "end_time": best_end.strftime("%H:%M"),
            "parking_position": parking,
            "delay_min": round(best_metric[0], 2)
        })

        # Remove only this aircraft’s task
        ready_tasks.remove(best_task)
        # Next task can’t start before this one ends
        current_time = best_end

# === Save results ===
schedule_df = pd.DataFrame(assignments)
schedule_df.to_csv("optimized_schedule.csv", index=False)
print("✅ Optimized schedule saved: optimized_schedule.csv")

# === Compute delays ===
aircraft_results = []
for ac_id, ac in aircraft_info.items():
    planned_end = ac["planned_end"]
    group = schedule_df[schedule_df["aircraft_id"] == ac_id]
    if not group.empty:
        last_end = parse_time(group["end_time"].max())
    else:
        last_end = planned_end
    effective_end = max(last_end, planned_end)
    delay = max(0, (effective_end - planned_end).total_seconds() / 60)
    aircraft_results.append({
        "aircraft_id": ac_id,
        "planned_end": planned_end.strftime("%H:%M"),
        "optimized_end": effective_end.strftime("%H:%M"),
        "optimized_delay_min": round(delay, 2)
    })

pd.DataFrame(aircraft_results).to_csv("aircraft_results.csv", index=False)
print("✅ Aircraft results saved: aircraft_results.csv")
