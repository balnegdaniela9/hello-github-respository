import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import MODEL_FEATURES, TARGET_COLUMN
from modules.prediction import predict_batch

def show_dashboard(df, model=None):
    st.subheader("Monitoring Dashboard")
    total = len(df)
    historical_breaches = None
    if TARGET_COLUMN in df.columns:
        vals = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
        if vals.notna().any():
            historical_breaches = int((vals == 1).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Requests in uploaded file", f"{total:,}")
    c2.metric("Historical labeled breaches", "Not available" if historical_breaches is None else f"{historical_breaches:,}")
    c3.metric("Saved model available", "Yes" if model is not None else "No")

    if "Priority" in df.columns:
        priority = df["Priority"].fillna("Missing").astype(str).value_counts().rename_axis("Priority").reset_index(name="Requests")
        st.plotly_chart(px.bar(priority, x="Priority", y="Requests", title="Requests by Priority"), use_container_width=True)

    if model is None:
        st.warning("Risk queue is unavailable until the fitted model is exported to models/trained_model.joblib.")
        st.dataframe(df[[c for c in ["Service_Type", "Priority", "Queue_Length", "Required_Documents_Complete", "SLA_Days"] if c in df.columns]].head(50), use_container_width=True)
        return

    try:
        results = predict_batch(model, df)
        st.markdown("### Staff review queue")
        queue = results[results["Predicted_SLA_Breached"].astype(str).isin(["1", "1.0", "True"])].copy()
        if "Estimated_Breach_Probability" in queue.columns:
            queue = queue.sort_values("Estimated_Breach_Probability", ascending=False)
        display_cols = [c for c in ["Request_ID", "Service_Type", "Priority", "Queue_Length", "Required_Documents_Complete", "SLA_Days", "Estimated_Breach_Probability", "Review_Recommendation"] if c in queue.columns]
        if queue.empty:
            st.success("The model did not flag any rows in this uploaded file. Continue normal monitoring; this does not guarantee there will be no breaches.")
        else:
            st.dataframe(queue[display_cols], use_container_width=True)
            st.download_button("Download review queue", queue[display_cols].to_csv(index=False).encode("utf-8"), "staff_review_queue.csv", "text/csv")
    except Exception as exc:
        st.error(f"Could not build the risk queue: {exc}")

    st.info("This queue is advisory. Authorized staff must verify request details and decide on appropriate follow-up.")
