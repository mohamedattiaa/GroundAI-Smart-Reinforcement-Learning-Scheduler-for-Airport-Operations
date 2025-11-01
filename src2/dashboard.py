import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from pathlib import Path
import io

st.set_page_config(page_title="Airport Optimization Dashboard", layout="wide")
data_dir = Path("data")

st.title("💼 Airport Ground Operations Optimizer")
st.markdown("### Upload your own dataset and visualize optimization results")

# === File Upload Section ===
st.sidebar.header("📁 Upload Dataset (Optional)")
uploaded_files = st.sidebar.file_uploader(
    "Upload aircraft.csv, tasks.csv, vehicles.csv, travel_time.csv",
    type=["csv"],
    accept_multiple_files=True
)

if uploaded_files:
    for file in uploaded_files:
        with open(data_dir / file.name, "wb") as f:
            f.write(file.getbuffer())
    st.sidebar.success("✅ Files uploaded successfully!")
    st.sidebar.write("You can now rerun the optimizer using your uploaded data.")

# === Load Results ===
try:
    schedule = pd.read_csv("optimized_schedule.csv")
    aircraft_summary = pd.read_csv("aircraft_results.csv")
except FileNotFoundError:
    st.error("Please run heuristic_optimizer.py first to generate or upload data.")
    st.stop()

# === Gantt Chart ===
def plot_gantt(schedule):
    schedule["start_dt"] = schedule["start_time"].apply(lambda x: datetime.strptime(x, "%H:%M"))
    schedule["end_dt"] = schedule["end_time"].apply(lambda x: datetime.strptime(x, "%H:%M"))
    vehicles = schedule["vehicle_id"].unique()
    colors = {v: plt.cm.tab20(i / len(vehicles)) for i, v in enumerate(vehicles)}
    y_positions = {ac: i for i, ac in enumerate(sorted(schedule["aircraft_id"].unique()))}
    fig, ax = plt.subplots(figsize=(12, 5))
    for _, row in schedule.iterrows():
        y = y_positions[row["aircraft_id"]]
        start = row["start_dt"]
        end = row["end_dt"]
        color = colors[row["vehicle_id"]]
        ax.barh(y, (end - start).seconds / 60,
                left=(start - schedule["start_dt"].min()).seconds / 60,
                color=color, edgecolor="black", height=0.5)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()))
    ax.set_xlabel("Minutes since start")
    ax.set_ylabel("Aircraft")
    ax.set_title("🛫 Optimized Turnaround Gantt Chart")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    patches = [mpatches.Patch(color=colors[v], label=v) for v in vehicles]
    ax.legend(handles=patches, title="Vehicles", loc="upper right")
    st.pyplot(fig)

# === Vehicle Path Map ===
def plot_vehicle_paths(schedule):
    schedule["pos_num"] = schedule["parking_position"].str.extract(r'(\d+)').astype(int)
    vehicles = schedule["vehicle_id"].unique()
    colors = {v: plt.cm.tab10(i / len(vehicles)) for i, v in enumerate(vehicles)}
    positions = sorted(schedule["pos_num"].unique())
    y_map = {p: i for i, p in enumerate(positions)}
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in positions:
        ax.scatter(0, y_map[p], s=150, color="skyblue", edgecolor="black")
        ax.text(0.2, y_map[p], f"P{p}", va="center", fontsize=9)
    for v in vehicles:
        v_df = schedule[schedule["vehicle_id"] == v].sort_values("start_time")
        times = list(range(len(v_df)))
        ax.plot(times, v_df["pos_num"].map(y_map), marker="o", color=colors[v], label=v)
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels([f"P{p}" for p in positions])
    ax.set_xlabel("Task Sequence")
    ax.set_ylabel("Parking Positions")
    ax.set_title("🚚 Vehicle Movement Between Parking Stands")
    ax.legend(title="Vehicles", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)
    st.pyplot(fig)

# === Dashboard Tabs ===
tab1, tab2, tab3 = st.tabs(["📊 Gantt Chart", "📈 Aircraft Summary", "🗺️ Vehicle Paths"])

with tab1:
    st.header("📊 Gantt Chart")
    plot_gantt(schedule)

with tab2:
    st.header("📈 Aircraft Turnaround Summary")
    st.dataframe(aircraft_summary, use_container_width=True)

with tab3:
    st.header("🗺️ Vehicle Movement Map")
    plot_vehicle_paths(schedule)
