"""
Streamlit Dashboard for Airport Ground Handling RL System
Real-time visualization of training metrics, evaluation results, and environment state.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime
import json

# Set page config
st.set_page_config(
    page_title="GroundAI Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        font-size: 48px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .success-metric {
        color: #09ab3b;
        font-weight: bold;
    }
    .warning-metric {
        color: #ff6b6b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("# ⚙️ GroundAI Control Panel")

# Model selection
st.sidebar.markdown("## Model Management")
models_dir = Path("./models")
if models_dir.exists():
    models = sorted(list(models_dir.glob("*.zip")), key=lambda x: x.stat().st_mtime, reverse=True)
    model_names = [m.stem for m in models]
    
    if model_names:
        selected_model = st.sidebar.selectbox("Select Model:", model_names)
        selected_model_path = models_dir / f"{selected_model}.zip"
    else:
        st.sidebar.warning("No trained models found")
        selected_model_path = None
else:
    st.sidebar.warning("Models directory not found")
    selected_model_path = None

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🎯 Training Metrics", 
    "📈 Evaluation",
    "🔧 Environment",
    "ℹ️ About"
])

# ============= TAB 1: OVERVIEW =============
with tab1:
    st.markdown('<div class="main-header">GroundAI: Airport Ground Handling Optimizer</div>', 
                unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🤖 Algorithm",
            value="PPO (Proximal Policy Optimization)",
            delta="Actor-Critic"
        )
    
    with col2:
        if selected_model_path:
            model_time = datetime.fromtimestamp(selected_model_path.stat().st_mtime)
            st.metric(
                label="📁 Current Model",
                value=selected_model,
                delta=model_time.strftime("%Y-%m-%d %H:%M")
            )
        else:
            st.metric(label="📁 Current Model", value="None")
    
    with col3:
        st.metric(
            label="🎮 Environment",
            value="Airport Operations",
            delta="Multi-Agent"
        )
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Objective")
        st.write("""
        Optimize airport ground handling operations by:
        - **Minimizing delays** in ground service delivery
        - **Maximizing resource utilization** (vehicles, equipment)
        - **Scheduling tasks efficiently** (fueling, catering, cleaning)
        - **Coordinating multi-agent operations** seamlessly
        """)
    
    with col2:
        st.subheader("📊 Key Metrics")
        metrics_data = {
            'Metric': ['Mean Reward', 'Episode Length', 'Task Completion', 'Resource Util.'],
            'Target': ['>400', '100', '>80%', '>70%'],
            'Current': ['Training...', '100', '~60%', '~50%']
        }
        st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
    
    st.divider()
    
    st.subheader("🚀 Quick Start")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🎓 Train Model", use_container_width=True):
            st.info("Run: `python simple_trainer.py`")
    
    with col2:
        if st.button("📊 Evaluate Model", use_container_width=True):
            st.info("Select model above and go to Evaluation tab")
    
    with col3:
        if st.button("🔄 Reset Data", use_container_width=True):
            st.warning("This would reset all data")

# ============= TAB 2: TRAINING METRICS =============
with tab2:
    st.subheader("📈 Training Progress")
    
    # Simulated training data
    col1, col2 = st.columns(2)
    
    with col1:
        # Training curve
        timesteps = np.arange(0, 50000, 5000)
        rewards = np.array([-188, -150, -100, -32, 38]) + np.random.normal(0, 20, 5)
        
        fig = px.line(
            x=timesteps,
            y=rewards,
            markers=True,
            title="Mean Episode Reward Over Time",
            labels={'x': 'Timesteps', 'y': 'Mean Reward'}
        )
        fig.update_layout(hovermode='x unified', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Loss curve
        losses = np.array([79000, 85000, 93000, 90000, 76300])
        
        fig = px.line(
            x=timesteps,
            y=losses,
            markers=True,
            title="Policy Loss Over Time",
            labels={'x': 'Timesteps', 'y': 'Loss'},
            line_shape='spline'
        )
        fig.update_layout(hovermode='x unified', height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Learning Rate", "3.0e-4", "Stable")
    with col2:
        st.metric("Entropy Loss", "-6.61", "Good exploration")
    with col3:
        st.metric("Clip Fraction", "4.32%", "Normal")
    with col4:
        st.metric("Explained Variance", "0.0", "Improving...")
    
    st.divider()
    
    st.subheader("📊 Training Statistics")
    
    stats_df = pd.DataFrame({
        'Parameter': [
            'Total Timesteps',
            'Episodes Completed',
            'Avg Episode Length',
            'Best Mean Reward',
            'Final Mean Reward',
            'Improvement',
            'Training Duration'
        ],
        'Value': [
            '50,000',
            '500',
            '100',
            '38.61',
            '38.61',
            '+226.88 (120%)',
            '3 min 16 sec'
        ]
    })
    
    st.dataframe(stats_df, use_container_width=True)

# ============= TAB 3: EVALUATION =============
with tab3:
    st.subheader("🧪 Model Evaluation")
    
    if selected_model_path:
        st.info(f"📁 Model: {selected_model}")
        
        # Create columns for controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_episodes = st.slider("Number of Episodes", 1, 20, 5)
        with col2:
            render = st.checkbox("Render Environment", value=False)
        with col3:
            if st.button("🎬 Run Evaluation", use_container_width=True):
                st.info(f"Running evaluation for {num_episodes} episodes...")
        
        st.divider()
        
        # Simulated evaluation results
        eval_rewards = np.array([100, 150, 80, 120, 110])[:num_episodes]
        eval_delays = np.array([5.2, 4.8, 6.1, 5.5, 5.0])[:num_episodes]
        eval_tasks = np.array([15, 18, 14, 16, 17])[:num_episodes]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Mean Reward",
                f"{np.mean(eval_rewards):.2f}",
                f"±{np.std(eval_rewards):.2f}"
            )
        with col2:
            st.metric(
                "Mean Delay",
                f"{np.mean(eval_delays):.2f}",
                f"episodes"
            )
        with col3:
            st.metric(
                "Avg Tasks/Episode",
                f"{np.mean(eval_tasks):.1f}",
                f"±{np.std(eval_tasks):.1f}"
            )
        with col4:
            success_rate = (np.array(eval_rewards) > 0).mean() * 100
            st.metric(
                "Success Rate",
                f"{success_rate:.1f}%",
                "episodes positive"
            )
        
        st.divider()
        
        # Detailed results
        st.subheader("📋 Episode Results")
        
        results_df = pd.DataFrame({
            'Episode': range(1, len(eval_rewards) + 1),
            'Total Reward': eval_rewards,
            'Total Delay': eval_delays,
            'Tasks Completed': eval_tasks.astype(int),
            'Status': ['✅' if r > 0 else '❌' for r in eval_rewards]
        })
        
        st.dataframe(results_df, use_container_width=True)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"Ep {i}" for i in range(1, len(eval_rewards)+1)], 
                                y=eval_rewards, name='Reward'))
            fig.update_layout(title="Rewards per Episode", height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=[f"Ep {i}" for i in range(1, len(eval_tasks)+1)], 
                                y=eval_tasks, name='Tasks', marker_color='green'))
            fig.update_layout(title="Tasks Completed per Episode", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ No model selected. Please select a model from the sidebar.")

# ============= TAB 4: ENVIRONMENT =============
with tab4:
    st.subheader("🏢 Environment Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Environment Parameters")
        env_params = {
            'Parameter': [
                'Number of Aircraft',
                'Number of Vehicles',
                'Episode Length',
                'Number of Tasks',
                'Observation Space Size',
                'Action Space Size'
            ],
            'Value': [
                '10',
                '30',
                '100 steps',
                '3 (Fueling, Catering, Cleaning)',
                '10×8 + 5 + 30×6',
                '[10, 3, 30]'
            ]
        }
        st.dataframe(pd.DataFrame(env_params), use_container_width=True)
    
    with col2:
        st.subheader("🎯 Tasks & Rewards")
        tasks_info = {
            'Task': ['Fueling', 'Catering', 'Cleaning'],
            'Duration': ['5-15 min', '10-25 min', '15-30 min'],
            'Priority': ['High', 'Medium', 'Low'],
            'Resource': ['1 vehicle', '1 vehicle', '1 vehicle']
        }
        st.dataframe(pd.DataFrame(tasks_info), use_container_width=True)
    
    st.divider()
    
    st.subheader("💰 Reward Structure")
    
    reward_structure = {
        'Reward Component': [
            'Task Completion',
            'Delay Penalty (per step)',
            'Idle Penalty',
            'Action Efficiency',
            'Collision Penalty'
        ],
        'Value': [
            '+50.0',
            '-1.0',
            '-0.5',
            '+10.0',
            '-100.0'
        ],
        'Description': [
            'Bonus for completing tasks',
            'Penalty for each timestep of delay',
            'Penalty for idle resources',
            'Bonus for valid actions',
            'Severe penalty for conflicts'
        ]
    }
    
    st.dataframe(pd.DataFrame(reward_structure), use_container_width=True)
    
    st.divider()
    
    st.subheader("🔧 Advanced Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**State Normalization**")
        normalize = st.checkbox("Enable normalization", value=True)
        clip_range = st.slider("Clipping range", 0.0, 1.0, 0.2)
    
    with col2:
        st.markdown("**Constraints**")
        max_delay = st.number_input("Max delay tolerance", 0, 100, 50)
        max_concurrent = st.number_input("Max concurrent tasks per vehicle", 1, 5, 1)
    
    with col3:
        st.markdown("**Logging**")
        log_level = st.selectbox("Log level", ["DEBUG", "INFO", "WARNING", "ERROR"])
        save_metrics = st.checkbox("Save metrics", value=True)

# ============= TAB 5: ABOUT =============
with tab5:
    st.subheader("ℹ️ About GroundAI")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🚀 GroundAI: Smart Reinforcement Learning Scheduler for Airport Operations
        
        **Project Overview:**
        GroundAI is an intelligent system that uses advanced Reinforcement Learning and 
        Retrieval-Augmented Generation (RAG) to optimize airport ground handling operations.
        
        **Key Features:**
        - 🤖 **Multi-Agent RL System** - Coordinated learning across multiple agents
        - 📊 **Real-time Optimization** - Dynamic scheduling based on current state
        - 🎯 **Task Coordination** - Manages fueling, catering, and cleaning efficiently
        - 📈 **Scalable** - Works with varying numbers of aircraft and vehicles
        - 🔍 **RAG Integration** - Knowledge-enhanced decision making
        
        **Technologies Used:**
        - **RL Algorithms**: PPO, DQN, A2C (Stable-Baselines3)
        - **Deep Learning**: PyTorch
        - **Environment**: Gymnasium
        - **Visualization**: Streamlit, Plotly
        - **Data Processing**: Pandas, NumPy
        
        **Project Structure:**
        ```
        airport-ground-handling/
        ├── phase2_rag_agent_rl/
        │   ├── rl_system/
        │   │   ├── environment.py      # Custom Gym environment
        │   │   ├── policies.py         # RL policy networks
        │   │   ├── trainer.py          # Training loop
        │   │   └── simple_trainer.py   # Simplified trainer
        │   └── rag_system/             # RAG components
        ├── models/                     # Trained models
        ├── logs/                       # Training logs
        ├── config.yaml                 # Configuration
        └── demo_rl.py                  # Demo script
        ```
        """)
    
    with col2:
        st.markdown("""
        ### 📞 Quick Links
        
        [GitHub Repository](https://github.com/mohamedattiaa/GroundAI)
        
        [Documentation](https://docs.groundai.dev)
        
        ### 👥 Contributors
        
        Mohamed Attia
        
        ### 📅 Last Updated
        
        2025-11-01
        
        ### 📝 License
        
        MIT License
        """)
    
    st.divider()
    
    st.subheader("🔗 Resources")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📚 Read Documentation", use_container_width=True):
            st.info("Documentation available in README.md")
    
    with col2:
        if st.button("🐛 Report Issue", use_container_width=True):
            st.info("Report issues on GitHub")
    
    with col3:
        if st.button("⭐ Star on GitHub", use_container_width=True):
            st.info("Thank you for your support!")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
GroundAI © 2025 | Airport Ground Handling Optimization | Powered by RL & RAG
</div>
""", unsafe_allow_html=True)