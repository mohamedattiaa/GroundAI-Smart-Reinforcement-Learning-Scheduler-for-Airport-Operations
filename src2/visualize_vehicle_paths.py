import pandas as pd
import matplotlib.pyplot as plt

# === Load ===
df = pd.read_csv("optimized_schedule.csv")

# Extract numeric positions
df["from_pos"] = df["parking_position"].str.extract(r'(\d+)').astype(int)

# Prepare figure
fig, ax = plt.subplots(figsize=(8, 6))
positions = sorted(df["from_pos"].unique())
y_map = {p: i for i, p in enumerate(positions)}

# Draw parking stand positions
for p in positions:
    ax.scatter(0, y_map[p], s=200, label=f"P{p}", color="skyblue", edgecolor="black")
    ax.text(0.1, y_map[p], f"P{p}", va="center", fontsize=10)

# Draw vehicle paths (simplified)
vehicles = df["vehicle_id"].unique()
colors = {v: plt.cm.tab10(i / len(vehicles)) for i, v in enumerate(vehicles)}

for v in vehicles:
    v_df = df[df["vehicle_id"] == v].sort_values("start_time")
    times = range(len(v_df))
    ax.plot(times, v_df["from_pos"].map(y_map), marker="o", color=colors[v], label=v)
    for _, r in v_df.iterrows():
        ax.text(times.__iter__().__next__(), y_map[r["from_pos"]], v, fontsize=7, color=colors[v])

ax.set_yticks(list(y_map.values()))
ax.set_yticklabels([f"P{p}" for p in positions])
ax.set_xlabel("Task Sequence (per vehicle)")
ax.set_ylabel("Parking Positions")
ax.set_title("🚚 Vehicle Movement Between Parking Stands")
ax.legend(title="Vehicles", loc="upper right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
