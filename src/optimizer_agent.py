# === Autonomous Optimizer Agent ===
# Reads dataset, optimizes assignments automatically (no user command needed)
import pandas as pd
from ortools.linear_solver import pywraplp

def optimize_assignments():
    print("🧮 Optimizing vehicle-task assignments...")

    tasks = pd.read_csv("data/tasks.csv")
    vehicles = pd.read_csv("data/vehicles.csv")

    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        print("Error: OR-Tools solver not available.")
        return

    # Decision vars: x[i,j] = 1 if vehicle j assigned to task i
    x = {}
    for i, t in tasks.iterrows():
        for j, v in vehicles.iterrows():
            if t["required_vehicle_type"] == v["vehicle_type"]:
                x[(i,j)] = solver.BoolVar(f"x_{i}_{j}")

    # Objective: minimize total delay + travel time
    solver.Minimize(
        sum(x[(i,j)] * (t["duration_min"] + 0.1) for (i,j), t in zip(x.keys(), tasks.iloc[[i for (i,j) in x.keys()]].itertuples()))
    )

    # Each task assigned to exactly one vehicle
    for i, t in tasks.iterrows():
        solver.Add(sum(x[(i,j)] for j, v in enumerate(vehicles.itertuples()) if (i,j) in x) == 1)

    # Solve
    status = solver.Solve()
    if status == pywraplp.Solver.OPTIMAL:
        print("✅ Optimal assignment found.\n")
        results = []
        for (i,j), var in x.items():
            if var.solution_value() > 0.5:
                results.append({
                    "task_id": tasks.iloc[i]["task_id"],
                    "vehicle_id": vehicles.iloc[j]["vehicle_id"],
                    "aircraft_id": tasks.iloc[i]["aircraft_id"]
                })
        df_out = pd.DataFrame(results)
        df_out.to_csv("data/optimized_assignments.csv", index=False)
        print(df_out)
    else:
        print("⚠️ No optimal solution found.")

if __name__ == "__main__":
    optimize_assignments()
