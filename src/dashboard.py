import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from chat_rag_memory import qa_chain
from optimizer_agent import optimize_assignments

st.set_page_config(page_title="Airport AI Dashboard", layout="wide")

st.title("🛫 Airport Operations AI Dashboard")

tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "📈 Gantt Chart", "⚙️ Optimizer"])

with tab1:
    st.header("💬 Chat with AI (Memory + RAG)")
    user_input = st.text_input("Ask a question:")
    if st.button("Ask"):
        answer = qa_chain.run(user_input)
        st.write(f"**AI:** {answer}")

with tab2:
    st.header("📊 Visualize Schedule")
    df = pd.read_csv("data/assignments.csv")
    st.dataframe(df)

with tab3:
    st.header("⚙️ Run Optimizer")
    if st.button("Run Optimization"):
        optimize_assignments()
        st.success("✅ Optimization complete! Check optimized_assignments.csv")
