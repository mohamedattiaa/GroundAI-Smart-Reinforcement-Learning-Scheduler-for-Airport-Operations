# comparison_dashboard.py
# Streamlit comparison dashboard for:
#   - baseline_schedule.csv (naive baseline)
#   - optimized_schedule.csv (heuristic optimizer)
# also shows per-aircraft planned vs baseline vs optimized finish times
# and vehicle movement timelines (including transit segments)
#
# Run:
#   streamlit run comparison_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
# Consistent task colors across both charts
TASK_COLORS = {
    "Deboarding": "#1f77b4",
    "Cleaning": "#aec7e8",
    "Catering": "#ff9896",
    "Refueling": "#d62728",
    "Boarding": "#2ca02c",
    "Pushback": "#9467bd",
    "Baggage": "#8c564b",
}


st.set_page_config(page_title="Schedule Comparison", layout="wide")
st.title("🛫 Airport Scheduling — Baseline vs Optimized (Comparison)")

# ----------------------------
# Helper utilities
# ----------------------------
def safe_read_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return None

def to_dt_series(df, col):
    """Convert df[col] to pandas datetime (naive local) if present."""
    if col in df.columns:
        return pd.to_datetime(df[col], format="%H:%M", errors="coerce")
    return None

def ensure_datetime_series(df, col):
    s = to_dt_series(df, col)
    if s is None:
        return None
    # Convert to python datetimes to avoid Plotly vline issues later when extracting .max()
    return s.dt.to_pydatetime()

def add_finish_vline(fig, finish_dt, label="Finish"):
    # finish_dt should be a python datetime (not pandas.Timestamp)
    if finish_dt is None:
        return
    if isinstance(finish_dt, pd.Timestamp):
        finish_dt = finish_dt.to_pydatetime()
    # add vertical line and annotation using go layout shapes and annotations (works reliably)
    fig.add_vline(x=finish_dt, line_width=2, line_dash="dash", line_color="red")
    # add annotation
    fig.add_annotation(
        x=finish_dt,
        y=1.02,
        xref="x",
        yref="paper",
        text=f"{label}: {finish_dt.strftime('%H:%M')}",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.8)",
        bordercolor="red",
    )

# ----------------------------
# Load schedules
# ----------------------------
naive_df = safe_read_csv("baseline_schedule.csv")
opt_df = safe_read_csv("optimized_schedule.csv")
aircraft_df = safe_read_csv("data/aircraft.csv")
travel_df = safe_read_csv("data/travel_time.csv")

if naive_df is None or opt_df is None:
    st.error("Missing schedule files. Please run baseline_scheduler.py and heuristic_optimizer.py first to generate:")
    st.write("- baseline_schedule.csv")
    st.write("- optimized_schedule.csv")
    st.stop()

# Ensure time columns are parsed
for df in (naive_df, opt_df):
    # accept a variety of column names (start_time/end_time or start/end)
    if "start_time" not in df.columns and "start" in df.columns:
        df.rename(columns={"start": "start_time"}, inplace=True)
    if "end_time" not in df.columns and "end" in df.columns:
        df.rename(columns={"end": "end_time"}, inplace=True)
    df["start_time_dt"] = pd.to_datetime(df["start_time"], format="%H:%M", errors="coerce")
    df["end_time_dt"] = pd.to_datetime(df["end_time"], format="%H:%M", errors="coerce")

# compute overall finish datetimes (python datetimes)
naive_finish_ts = naive_df["end_time_dt"].max()
opt_finish_ts = opt_df["end_time_dt"].max()
naive_finish = naive_finish_ts.to_pydatetime() if pd.notna(naive_finish_ts) else None
opt_finish = opt_finish_ts.to_pydatetime() if pd.notna(opt_finish_ts) else None

# ----------------------------
# Top metrics
# ----------------------------
st.subheader("🏁 Overall completion times")
c1, c2 = st.columns(2)
c1.metric("Baseline finish", naive_finish.strftime("%H:%M") if naive_finish else "N/A")
c2.metric("Optimized finish", opt_finish.strftime("%H:%M") if opt_finish else "N/A")

# ----------------------------
# Side-by-side Gantt charts
# ----------------------------
st.subheader("📊 Gantt charts (Baseline vs Optimized)")

left_col, right_col = st.columns(2)

with left_col:
    st.markdown("**Baseline (naive)**")
    hover_cols_naive = [c for c in ["vehicle_id", "parking_position", "task_name"] if c in naive_df.columns]
    fig_naive = px.timeline(
    naive_df,
    x_start="start_time_dt",
    x_end="end_time_dt",
    y="aircraft_id",
    color="task_name" if "task_name" in naive_df.columns else None,
    hover_data=hover_cols_naive,
    color_discrete_map=TASK_COLORS,
)

    fig_naive.update_yaxes(title="Aircraft")
    fig_naive.update_layout(height=500, xaxis_title="Time")
    # add finish line
    add_finish_vline(fig_naive, naive_finish, label="Baseline finish")
    st.plotly_chart(fig_naive, use_container_width=True)

with right_col:
    st.markdown("**Optimized (event-driven)**")
    hover_cols_opt = [c for c in ["vehicle_id", "parking_position", "task_name"] if c in opt_df.columns]
    fig_opt = px.timeline(
    opt_df,
    x_start="start_time_dt",
    x_end="end_time_dt",
    y="aircraft_id",
    color="task_name" if "task_name" in opt_df.columns else None,
    hover_data=hover_cols_opt,
    color_discrete_map=TASK_COLORS,
)

    fig_opt.update_yaxes(title="Aircraft")
    fig_opt.update_layout(height=500, xaxis_title="Time")
    add_finish_vline(fig_opt, opt_finish, label="Optimized finish")
    st.plotly_chart(fig_opt, use_container_width=True)

# ----------------------------
# Comparison table: planned vs baseline vs optimized finish times
# ----------------------------
st.subheader("📋 Aircraft finish times comparison")

# get planned end times from aircraft.csv if available
planned_map = {}
if aircraft_df is not None and "aircraft_id" in aircraft_df.columns and "end_time_planned" in aircraft_df.columns:
    for _, r in aircraft_df.iterrows():
        planned_map[r["aircraft_id"]] = r["end_time_planned"]

# baseline per-aircraft finish
baseline_results = []
for aid, g in naive_df.groupby("aircraft_id"):
    last_end = g["end_time_dt"].max()
    baseline_results.append((aid, last_end.to_pydatetime() if pd.notna(last_end) else None))

# optimized per-aircraft finish
opt_results = []
for aid, g in opt_df.groupby("aircraft_id"):
    last_end = g["end_time_dt"].max()
    opt_results.append((aid, last_end.to_pydatetime() if pd.notna(last_end) else None))

# Build comparison table
aircraft_ids = sorted(set([r[0] for r in baseline_results] + [r[0] for r in opt_results] + list(planned_map.keys())))
rows = []
for aid in aircraft_ids:
    planned = planned_map.get(aid, None)
    base_end = next((t for a, t in baseline_results if a == aid), None)
    opt_end = next((t for a, t in opt_results if a == aid), None)
    # compute delays relative to planned (minutes)
    def minutes_diff(a, b):
        if a is None or b is None:
            return None
        return int((a - b).total_seconds() / 60)
    # convert planned to python datetime if string
    planned_dt = None
    if planned:
        try:
            planned_dt = datetime.strptime(planned, "%H:%M")
        except:
            planned_dt = None
    base_delay = minutes_diff(base_end, planned_dt) if planned_dt else None
    opt_delay = minutes_diff(opt_end, planned_dt) if planned_dt else None
    rows.append({
        "aircraft_id": aid,
        "planned_end": planned_dt.strftime("%H:%M") if planned_dt else planned,
        "baseline_end": base_end.strftime("%H:%M") if base_end else None,
        "baseline_delay_min": base_delay,
        "optimized_end": opt_end.strftime("%H:%M") if opt_end else None,
        "optimized_delay_min": opt_delay
    })

comp_df = pd.DataFrame(rows)
st.dataframe(comp_df, use_container_width=True)

# ----------------------------
# Vehicle movement timeline and travel times
# ----------------------------
st.subheader("🚚 Vehicle movement timeline (optimized)")

def build_vehicle_movements(df, travel_df=None):
    # Input df must contain vehicle_id, start_time_dt, end_time_dt, parking_position (or infer)
    movements = []
    if "vehicle_id" not in df.columns:
        return pd.DataFrame(movements)
    df_sorted = df.sort_values(by=["vehicle_id", "start_time_dt"])
    for vid, grp in df_sorted.groupby("vehicle_id"):
        grp = grp.reset_index(drop=True)
        for i in range(len(grp)):
            row = grp.loc[i]
            pos = row.get("parking_position", None)
            movements.append({
                "vehicle_id": vid,
                "segment_type": "Task",
                "parking": pos if pos is not None else (row.get("parking_position") if "parking_position" in row else None),
                "start": row["start_time_dt"],
                "end": row["end_time_dt"],
                "task": row.get("task_name", "")
            })
            # transit to next task if exists
            if i < len(grp) - 1:
                next_row = grp.loc[i+1]
                t_end = row["end_time_dt"]
                t_next_start = next_row["start_time_dt"]
                if pd.notna(t_end) and pd.notna(t_next_start) and t_next_start > t_end:
                    # compute travel minutes if travel_df and parking positions available
                    p1 = row.get("parking_position", None)
                    p2 = next_row.get("parking_position", None)
                    travel_min = None
                    if travel_df is not None and p1 is not None and p2 is not None:
                        # travel_df index/columns might be "P1" etc. Ensure strings
                        try:
                            travel_min = int(travel_df.loc[str(p1), str(p2)])
                        except Exception:
                            # try stripping 'P'
                            try:
                                travel_min = int(travel_df.loc[f"P{int(str(p1).strip('P'))}", f"P{int(str(p2).strip('P'))}"])
                            except Exception:
                                travel_min = int((t_next_start - t_end).total_seconds() / 60)
                    else:
                        travel_min = int((t_next_start - t_end).total_seconds() / 60)
                    movements.append({
                        "vehicle_id": vid,
                        "segment_type": "Transit",
                        "parking": f"{p1}→{p2}" if p1 and p2 else "Transit",
                        "start": t_end,
                        "end": t_next_start,
                        "task": f"Transit ({travel_min} min)"
                    })
    return pd.DataFrame(movements)

veh_moves = build_vehicle_movements(opt_df, travel_df)

if veh_moves.empty:
    st.info("No vehicle movement data found in optimized schedule.")
else:
    fig_vm = px.timeline(
        veh_moves,
        x_start="start",
        x_end="end",
        y="vehicle_id",
        color="segment_type",
        hover_data=["parking", "task"],
    )
    fig_vm.update_layout(height=450, xaxis_title="Time", yaxis_title="Vehicle")
    st.plotly_chart(fig_vm, use_container_width=True)

# ----------------------------
# Travel time matrix display
# ----------------------------
st.subheader("🗺 Travel time matrix (data/travel_time.csv)")
if travel_df is not None:
    try:
        display_df = travel_df.set_index(travel_df.columns[0]) if travel_df.columns[0] != travel_df.index.name else travel_df
        st.dataframe(travel_df, use_container_width=True)
    except Exception:
        st.dataframe(travel_df, use_container_width=True)
else:
    st.info("No travel_time.csv found in data/")

st.markdown("---")
st.markdown("**Notes:**\n- Ensure `baseline_schedule.csv` and `optimized_schedule.csv` exist in this folder.\n- This dashboard converts schedule times in `HH:MM` format to datetimes for plotting.")
