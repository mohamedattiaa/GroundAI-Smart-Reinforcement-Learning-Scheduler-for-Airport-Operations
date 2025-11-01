# src2/compare_debug.py
import pandas as pd
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data")
BASE = Path("baseline_schedule.csv")
OPT = Path("optimized_schedule.csv")
TRAVEL = DATA_DIR / "travel_time.csv"

def to_dt(s):
    return pd.to_datetime(s, format="%H:%M", errors="coerce")

def minutes(a,b):
    return (a - b).total_seconds()/60 if pd.notna(a) and pd.notna(b) else None

# load
naive = pd.read_csv(BASE)
opt = pd.read_csv(OPT)
travel = None
if TRAVEL.exists():
    try:
        travel = pd.read_csv(TRAVEL, index_col=0)
    except:
        travel = None

# unify names
for df in (naive, opt):
    if "start_time" not in df.columns and "start" in df.columns: df.rename(columns={"start":"start_time"}, inplace=True)
    if "end_time" not in df.columns and "end" in df.columns: df.rename(columns={"end":"end_time"}, inplace=True)
    df["start_dt"] = to_dt(df["start_time"])
    df["end_dt"]   = to_dt(df["end_time"])

print("Rows: baseline", len(naive), "optimized", len(opt))
print()

# 1) per-aircraft finish & delays
def per_aircraft(df):
    res=[]
    for ac, g in df.groupby("aircraft_id"):
        last_end = g["end_dt"].max()
        first_start = g["start_dt"].min()
        res.append((ac, first_start, last_end, len(g)))
    return pd.DataFrame(res, columns=["aircraft_id","first_start","last_end","n_tasks"])

naive_ac = per_aircraft(naive)
opt_ac = per_aircraft(opt)
print("Per-aircraft summary (baseline)")
print(naive_ac)
print()
print("Per-aircraft summary (optimized)")
print(opt_ac)
print()

# 2) Compare finishes side-by-side (requires data/aircraft.csv planned end)
planned_map = {}
aircraft_file = DATA_DIR / "aircraft.csv"
if aircraft_file.exists():
    acdf = pd.read_csv(aircraft_file)
    for _, r in acdf.iterrows():
        planned_map[r["aircraft_id"]] = r["end_time_planned"]
else:
    print("Warning: data/aircraft.csv not found; planned ends not available.")

comp = []
all_ac = sorted(set(naive_ac["aircraft_id"]) | set(opt_ac["aircraft_id"]) | set(planned_map.keys()))
for ac in all_ac:
    naive_last = naive_ac[naive_ac["aircraft_id"]==ac]["last_end"].iloc[0] if ac in list(naive_ac["aircraft_id"]) else None
    opt_last = opt_ac[opt_ac["aircraft_id"]==ac]["last_end"].iloc[0] if ac in list(opt_ac["aircraft_id"]) else None
    planned = planned_map.get(ac, None)
    try:
        planned_dt = pd.to_datetime(planned, format="%H:%M") if planned is not None else None
    except:
        planned_dt = None
    naive_delay = minutes(naive_last, planned_dt) if naive_last is not None and planned_dt is not None else None
    opt_delay   = minutes(opt_last, planned_dt) if opt_last is not None and planned_dt is not None else None
    comp.append({"aircraft_id":ac, "planned":planned, "baseline_end":naive_last, "baseline_delay":naive_delay, "opt_end":opt_last, "opt_delay":opt_delay})
compdf = pd.DataFrame(comp)
compdf.to_csv("debug_aircraft_comparison.csv", index=False)
print("Saved debug_aircraft_comparison.csv")
print(compdf)
print()

# 3) vehicle utilizations: total busy time, number of tasks, gaps (idle), inferred transit sum
def vehicle_stats(df):
    stats=[]
    for vid, g in df.sort_values(["vehicle_id","start_dt"]).groupby("vehicle_id"):
        g = g.reset_index(drop=True)
        busy = 0.0
        transit = 0.0
        overlaps = 0
        for i,row in g.iterrows():
            dur = minutes(row["end_dt"], row["start_dt"])
            busy += dur if dur is not None else 0
            if i>0:
                prev_end = g.loc[i-1,"end_dt"]
                gap = minutes(row["start_dt"], prev_end)
                if gap is None:
                    continue
                if gap < 0:
                    overlaps += 1
                else:
                    transit += gap
        stats.append({"vehicle_id":vid, "n_tasks":len(g), "busy_min":round(busy,2), "transit_min":round(transit,2), "overlaps":overlaps})
    return pd.DataFrame(stats)

vnaive = vehicle_stats(naive)
vopt = vehicle_stats(opt)
vnaive.to_csv("debug_vehicle_baseline.csv", index=False)
vopt.to_csv("debug_vehicle_optimized.csv", index=False)
print("Vehicle stats (baseline):")
print(vnaive)
print()
print("Vehicle stats (optimized):")
print(vopt)
print()

# 4) Check for impossible conditions:
# - tasks scheduled before aircraft arrival
# - overlapping tasks on same vehicle (overlaps>0)
errors=[]
# a) tasks before arrival
if aircraft_file.exists():
    arrivals = {r["aircraft_id"]:pd.to_datetime(r["start_time"], format="%H:%M") for _,r in acdf.iterrows()}
    for _, row in naive.iterrows():
        ac=row["aircraft_id"]
        st = row["start_dt"]
        if ac in arrivals and pd.notna(st) and st < arrivals[ac]:
            errors.append(f"Baseline: task {row.get('task_name')} for {ac} starts before arrival ({st} < {arrivals[ac]})")
    for _, row in opt.iterrows():
        ac=row["aircraft_id"]
        st = row["start_dt"]
        if ac in arrivals and pd.notna(st) and st < arrivals[ac]:
            errors.append(f"Optimized: task {row.get('task_name')} for {ac} starts before arrival ({st} < {arrivals[ac]})")

# b) overlapping tasks per vehicle already computed
for df, label in [(vnaive,"Baseline"), (vopt,"Optimized")]:
    for _, r in df.iterrows():
        if r["overlaps"]>0:
            errors.append(f"{label}: vehicle {r['vehicle_id']} has {r['overlaps']} overlapping tasks")

# write warnings
with open("debug_warnings.txt","w") as f:
    if errors:
        f.write("\n".join(errors))
    else:
        f.write("No issues detected.")

print("Warnings written to debug_warnings.txt")
print("\nErrors / Warnings (first 20):")
print("\n".join(errors[:20]) if errors else "No warnings")

# 5) Summary metrics
sum_naive_delay = compdf["baseline_delay"].dropna().sum()
sum_opt_delay   = compdf["opt_delay"].dropna().sum()
print()
print("Total baseline delay (sum per aircraft in min):", sum_naive_delay)
print("Total optimized delay (sum per aircraft in min):", sum_opt_delay)
print()
print("Detailed CSVs written: debug_aircraft_comparison.csv, debug_vehicle_baseline.csv, debug_vehicle_optimized.csv, debug_warnings.txt")
