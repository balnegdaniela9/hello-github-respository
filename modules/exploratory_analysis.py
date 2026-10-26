import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import TARGET_COLUMN

def show_eda(df: pd.DataFrame):
    st.subheader("Exploratory Data Analysis")
    st.markdown("### Descriptive statistics")
    st.dataframe(df.describe(include="all").transpose(), use_container_width=True)

    chart_options = [c for c in ["Service_Type", "Requester_Type", "Requested_Channel", "Priority", "Assigned_Office", "Submission_Weekday"] if c in df.columns]
    if chart_options:
        selected = st.selectbox("Choose a category for the count chart", chart_options)
        counts = df[selected].fillna("Missing").astype(str).value_counts().rename_axis(selected).reset_index(name="Count")
        fig = px.bar(counts, x=selected, y="Count", title=f"Requests by {selected}")
        st.plotly_chart(fig, use_container_width=True)

    if "Queue_Length" in df.columns:
        queue = pd.to_numeric(df["Queue_Length"], errors="coerce").dropna()
        if not queue.empty:
            fig = px.histogram(queue, x="Queue_Length", nbins=20, title="Queue Length at Submission")
            st.plotly_chart(fig, use_container_width=True)

    if "SLA_Days" in df.columns:
        sla = pd.to_numeric(df["SLA_Days"], errors="coerce").dropna()
        if not sla.empty:
            fig = px.histogram(sla, x="SLA_Days", nbins=15, title="Published SLA Days")
            st.plotly_chart(fig, use_container_width=True)

    if TARGET_COLUMN in df.columns:
        target = df[TARGET_COLUMN].dropna().astype(str).value_counts().rename_axis(TARGET_COLUMN).reset_index(name="Count")
        fig = px.bar(target, x=TARGET_COLUMN, y="Count", title="Historical SLA Breach Label Distribution")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("The uploaded dataset has no SLA_Breached column. Target distribution is therefore not shown.")
