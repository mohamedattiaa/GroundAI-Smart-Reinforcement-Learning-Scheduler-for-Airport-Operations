# === Synthetic Airport Ground Operations Dataset Generator ===
# Python 3.9 compatible, generates realistic CSVs

import random, pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# --- Setup ---
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
random.seed(42)

# --- Parameters ---
NUM_AIRCRAFT = 5
NUM_VEHICLES = 8
STANDS = [f"S{i}" for i in range(1, 6)]
VEHICLE_TYPES = ["RefuelTruck", "CateringTruck", "CleaningVan", "Bus"]

def rand_time(base="08:00", offset_min=0, max_add=180):
    base_dt = datetime.strptime(base, "%H:%M") + timedelta(minutes=offset_min)
    return (base_dt + timedelta(minutes=random.randint(0, max_add))).strftime("%H:%M")

def add_minutes(t, mins):
    return (datetime.strptime(t, "%H:%M") + timedelta(minutes=mins)).strftime("%H:%M")

# 1️⃣ Aircraft Table
aircraft = []
for i in range(NUM_AIRCRAFT):
    arr = rand_time()
    dep = add_minutes(arr, random.randint(60,120))
    aircraft.append({
        "aircraft_id": f"A{i+1}",
        "arrival_time": arr,
        "departure_time": dep,
        "stand_id": random.choice(STANDS),
        "aircraft_type": random.choice(["A320","B737","A321"])
    })
df_aircraft = pd.DataFrame(aircraft)

# 2️⃣ Vehicles Table
vehicles = []
for i in range(NUM_VEHICLES):
    vt = random.choice(VEHICLE_TYPES)
    vehicles.append({
        "vehicle_id": f"V{i+1}",
        "vehicle_type": vt,
        "capacity": random.randint(1,5),
        "current_position": random.choice(["Depot_1","Depot_2","Depot_3"]),
        "speed_kmh": random.randint(15,30),
        "available_from": rand_time("07:00",0,60)
    })
df_vehicles = pd.DataFrame(vehicles)

# 3️⃣ Tasks Table
task_templates = {
    "A320":["Deboarding","Refueling","Cleaning","Boarding"],
    "B737":["Deboarding","Catering","Cleaning","Boarding"],
    "A321":["Deboarding","Refueling","Catering","Cleaning","Boarding"]
}
tasks=[]
for ac in aircraft:
    ac_id = ac["aircraft_id"]
    base_time = datetime.strptime(ac["arrival_time"], "%H:%M")
    for idx,ttype in enumerate(task_templates[ac["aircraft_type"]]):
        dur = random.randint(10,25)
        est = (base_time + timedelta(minutes=idx*15)).strftime("%H:%M")
        let = (base_time + timedelta(minutes=idx*15+dur+10)).strftime("%H:%M")
        req_v = "Bus" if "board" in ttype.lower() else \
                "RefuelTruck" if "refuel" in ttype.lower() else \
                "CateringTruck" if "cater" in ttype.lower() else "CleaningVan"
        tasks.append({
            "task_id": f"T_{ac_id}_{idx+1}",
            "aircraft_id": ac_id,
            "task_type": ttype,
            "duration_min": dur,
            "required_vehicle_type": req_v,
            "earliest_start": est,
            "latest_end": let,
            "precedence": f"T_{ac_id}_{idx}" if idx>0 else None
        })
df_tasks = pd.DataFrame(tasks)

# 4️⃣ Assignment Table
assignments=[]
for t in tasks:
    vmatch = df_vehicles[df_vehicles.vehicle_type==t["required_vehicle_type"]]
    if vmatch.empty: continue
    vsel = vmatch.sample(1).iloc[0]
    start = t["earliest_start"]
    end = add_minutes(start, t["duration_min"])
    assignments.append({
        "assignment_id": f"A_{t['task_id']}",
        "vehicle_id": vsel["vehicle_id"],
        "task_id": t["task_id"],
        "aircraft_id": t["aircraft_id"],
        "start_time": start,
        "end_time": end,
        "travel_time_min": random.randint(2,8),
        "delay_min": random.choice([0,0,5])
    })
df_assign = pd.DataFrame(assignments)

# 5️⃣ Travel Time Matrix
travel=[]
locs = ["Depot_1","Depot_2","Depot_3"] + STANDS
for a in locs:
    for b in locs:
        if a==b: continue
        travel.append({
            "from_location":a,
            "to_location":b,
            "distance_m": random.randint(300,900),
            "travel_time_min": random.randint(3,9)
        })
df_travel = pd.DataFrame(travel)

# 💾 Save all
df_aircraft.to_csv(data_dir/"aircraft.csv",index=False)
df_vehicles.to_csv(data_dir/"vehicles.csv",index=False)
df_tasks.to_csv(data_dir/"tasks.csv",index=False)
df_assign.to_csv(data_dir/"assignments.csv",index=False)
df_travel.to_csv(data_dir/"travel_times.csv",index=False)

print("✅ Synthetic dataset generated in 'data/' folder:")
print(f"Aircraft: {df_aircraft.shape}, Vehicles: {df_vehicles.shape}, Tasks: {df_tasks.shape}, Assignments: {df_assign.shape}")
