# src2/heuristic_optimizer_event_driven.py
"""
Event-driven vehicle-centric greedy scheduler:
- Whenever a vehicle becomes free, it immediately picks the ready task (across ALL aircraft)
  that it can start the earliest (considering travel, predecessors, and aircraft arrival).
- Respects task precedence and aircraft arrival times.
- Reads CSVs from data/, writes optimized_schedule.csv and aircraft_results.csv.
"""

import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import math

DATA_DIR = Path("data")
OUTPUT_SCHEDULE = "optimized_schedule.csv"
OUTPUT_RESULTS = "aircraft_results.csv"

# ---------------- helpers ----------------
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

# ---------------- load data ----------------
aircraft_df = pd.read_csv(DATA_DIR / "aircraft.csv")
tasks_df = pd.read_csv(DATA_DIR / "tasks.csv")
vehicles_df = pd.read_csv(DATA_DIR / "vehicles.csv")
travel_df = pd.read_csv(DATA_DIR / "travel_time.csv", index_col=0)

# normalize columns
vehicles_df["compatible_tasks"] = vehicles_df["compatible_tasks"].apply(parse_list)
tasks_df["precedence"] = tasks_df["precedence"].apply(parse_list)
tasks_df["task_id"] = tasks_df["task_id"].astype(int)
tasks_df = tasks_df.sort_values("task_id").reset_index(drop=True)

# build task metadata
task_meta = {}
for _, row in tasks_df.iterrows():
    tid = int(row["task_id"])
    task_meta[tid] = {
        "name": row["task_name"],
        "dur": int(row["duration_min"]),
        "preds": row["precedence"] if isinstance(row["precedence"], list) else []
    }

# build successor lists
for tid, meta in task_meta.items():
    meta["succ"] = []
for tid, meta in task_meta.items():
    for p in meta["preds"]:
        if p in task_meta:
            task_meta[p]["succ"].append(tid)

# helper travel time (positions like "P1","P2"), fallback default
def get_travel_time(from_pos: str, to_pos: str) -> int:
    try:
        return int(travel_df.loc[from_pos, to_pos])
    except Exception:
        # fallback to Manhattan style if P0..Pn not in matrix
        def pos_number(s):
            if isinstance(s, str) and s.startswith("P"):
                try:
                    return int(s[1:])
                except:
                    return 0
            return 0
        return 3 * abs(pos_number(from_pos) - pos_number(to_pos))

# ---------------- initialize vehicles ----------------
vehicles = {}
# default vehicle start time = earliest aircraft arrival or 08:00
if len(aircraft_df):
    earliest_arrival = min(aircraft_df["start_time"].astype(str).tolist())
    default_start = to_dt(earliest_arrival)
else:
    default_start = to_dt("08:00")

for _, v in vehicles_df.iterrows():
    vid = v["vehicle_id"]
    vehicles[vid] = {
        "compatible": v["compatible_tasks"],
        "available_at": default_start,
        "position": "P0"  # depot
    }

# ---------------- scheduling state ----------------
# tasks remaining per aircraft; scheduled flag per aircraft+task
scheduled = {}
# store each task's start/end when scheduled
schedule_records = []

# for each aircraft, track arrival, planned end, current earliest availability of aircraft (after tasks)
aircraft_info = {}
for _, ac in aircraft_df.iterrows():
    ac_id = ac["aircraft_id"]
    aircraft_info[ac_id] = {
        "parking": f"P{ac['parking_position']}",
        "arrival": to_dt(ac["start_time"]),
        "planned_end": to_dt(ac["end_time_planned"]),
        "done_tasks": set(),
        "pred_end": {}  # end times per task when completed
    }
    # scheduled set
    scheduled[ac_id] = set()

# initial ready set: tasks whose preds are empty for each aircraft
def ready_tasks_global():
    ready = []
    for ac_id, info in aircraft_info.items():
        for tid, meta in task_meta.items():
            if tid in scheduled[ac_id]:
                continue
            # preds must be scheduled
            preds = meta["preds"]
            if all(p in scheduled[ac_id] for p in preds):
                # earliest allowed start for this task considering preds and aircraft arrival
                if preds:
                    pred_ends = [aircraft_info[ac_id]["pred_end"][p] for p in preds]
                    earliest_from_preds = max(pred_ends)
                else:
                    earliest_from_preds = info["arrival"]
                ready.append((ac_id, tid, earliest_from_preds))
    return ready

# number of total tasks
total_tasks = len(task_meta) * len(aircraft_df)
scheduled_count = 0

# main loop: event-driven by earliest vehicle available
# to avoid infinite loops, guard with iterations
iter_guard = 0
max_iter = total_tasks * 10 + 1000

while scheduled_count < total_tasks and iter_guard < max_iter:
    iter_guard += 1

    # pick vehicle that'll be available the earliest
    next_vid = None
    next_available_time = None
    for vid, v in vehicles.items():
        if next_vid is None or v["available_at"] < next_available_time:
            next_vid = vid
            next_available_time = v["available_at"]

    if next_vid is None:
        break

    v = vehicles[next_vid]

    # get global ready tasks
    ready = ready_tasks_global()

    # if no ready tasks now, advance time to the earliest event:
    # either next vehicle available (but we chose earliest), or next task readiness (pred finishes)
    if not ready:
        # find earliest pred end across all aircraft (a future time when some task becomes ready)
        future_times = []
        for ac_id, info in aircraft_info.items():
            # tasks not scheduled whose preds not yet done: find max(pred_end) among their preds
            for tid, meta in task_meta.items():
                if tid in scheduled[ac_id]:
                    continue
                preds = meta["preds"]
                if preds:
                    # pred end times exist only if preds scheduled; else we can't compute
                    pred_done = [p in aircraft_info[ac_id]["pred_end"] for p in preds]
                    if any(pred_done):
                        # take max of available pred_end values (some preds may be scheduled)
                        times = [aircraft_info[ac_id]["pred_end"][p] for p in preds if p in aircraft_info[ac_id]["pred_end"]]
                        if times:
                            future_times.append(max(times))
                else:
                    # no preds but not ready? should be handled earlier
                    future_times.append(info["arrival"])
        if future_times:
            # advance this vehicle to the earliest of (next_available_time, min(future_times))
            min_future = min(future_times)
            if v["available_at"] < min_future:
                v["available_at"] = min_future
                # loop will pick same or other vehicle next iteration
                continue
        # nothing to wait for -> break
        break

    # For this vehicle, find the ready task that yields minimal start time considering travel
    best_choice = None
    best_start = None
    best_end = None
    for ac_id, tid, earliest_allowed in ready:
        if tid in scheduled[ac_id]:
            continue
        # only consider tasks vehicle can do
        if tid not in v["compatible"]:
            continue
        ac_pos = aircraft_info[ac_id]["parking"]
        travel = get_travel_time(v["position"], ac_pos)
        arrival_time = v["available_at"] + timedelta(minutes=travel)
        start_time = max(arrival_time, earliest_allowed)
        end_time = start_time + timedelta(minutes=task_meta[tid]["dur"])
        # choose minimal start_time (then minimal end_time tie-break)
        if best_start is None or (start_time < best_start) or (start_time == best_start and end_time < best_end):
            best_choice = (ac_id, tid, ac_pos, start_time, end_time)
            best_start = start_time
            best_end = end_time

    if best_choice is None:
        # this vehicle can't do any ready tasks now; advance its available to next earliest ready time and continue
        # compute earliest_allowed among ready tasks even if incompatible, to wait
        earliest_needed = min([earliest for (_, _, earliest) in ready])
        if v["available_at"] < earliest_needed:
            v["available_at"] = earliest_needed
            continue
        else:
            # no compatible ready tasks at this moment - mark vehicle idle a small epsilon to avoid infinite loop
            v["available_at"] += timedelta(minutes=1)
            continue

    # assign the chosen task
    ac_id, tid, ac_pos, start_time, end_time = best_choice

    schedule_records.append({
        "aircraft_id": ac_id,
        "parking_position": ac_pos,
        "task_id": tid,
        "task_name": task_meta[tid]["name"],
        "vehicle_id": next_vid,
        "start_time": to_str(start_time),
        "end_time": to_str(end_time),
        "delay_task_min": round((start_time - earliest_allowed).total_seconds() / 60.0, 2)
    })

    # update vehicle
    vehicles[next_vid]["available_at"] = end_time
    vehicles[next_vid]["position"] = ac_pos

    # update aircraft info
    aircraft_info[ac_id]["done_tasks"].add(tid)
    aircraft_info[ac_id]["pred_end"][tid] = end_time

    scheduled[ac_id].add(tid)
    scheduled_count += 1

# end while

# save schedule
sched_df = pd.DataFrame(schedule_records)
sched_df = sched_df.sort_values(by=["start_time", "aircraft_id"]).reset_index(drop=True)
sched_df.to_csv(OUTPUT_SCHEDULE, index=False)

# per-aircraft summary: last end, force not earlier than planned
results = []
for ac_id, info in aircraft_info.items():
    planned = info["planned_end"]
    # last end among scheduled tasks for this aircraft
    subs = sched_df[sched_df["aircraft_id"] == ac_id]
    if not subs.empty:
        last_end = to_dt(subs["end_time"].max())
    else:
        last_end = info["arrival"]
    effective_end = max(last_end, planned)
    delay = round((effective_end - planned).total_seconds() / 60.0, 2)
    results.append({
        "aircraft_id": ac_id,
        "planned_end": to_str(planned),
        "optimized_end": to_str(effective_end),
        "delay_total_min": delay
    })

pd.DataFrame(results).to_csv(OUTPUT_RESULTS, index=False)

print("✅ Event-driven optimizer finished.")
print(f"Wrote: {OUTPUT_SCHEDULE} and {OUTPUT_RESULTS}")
