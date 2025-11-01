# src2/compare_metrics.py
import pandas as pd

baseline_sum = pd.read_csv("baseline_aircraft_results.csv")
opt_sum = pd.read_csv("aircraft_results.csv")

def totals(df, label):
    df["delay_min"] = df["delay_min"].astype(float)
    total = df["delay_min"].sum()
    mean = df["delay_min"].mean()
    print(f"{label}: total delay = {total} min, mean delay = {mean:.2f} min")

print("=== Comparison ===")
totals(baseline_sum, "Baseline")
totals(opt_sum, "Optimized")

merged = baseline_sum.merge(opt_sum, on="aircraft_id", suffixes=("_base","_opt"))
merged["delta_total"] = merged["delay_min_opt"] - merged["delay_min_base"]
print("\nPer-aircraft comparison:")
print(merged[["aircraft_id","delay_min_base","delay_min_opt","delta_total"]])
