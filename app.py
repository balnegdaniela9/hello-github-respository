from pathlib import Path
import pandas as pd
import streamlit as st

from modules.data_processing import (
    MODEL_FEATURES, TARGET_COLUMN, load_and_validate_data,
    get_data_quality_summary
)
from modules.exploratory_analysis import show_eda
from modules.prediction import load_trained_model, predict_request, predict_batch
from modules.dashboard import show_dashboard

st.set_page_config(
    page_title="Student Service Request Monitoring System",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Student Service Request Monitoring System")
st.caption("Decision-support tool for early review of possible SLA breaches")

with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Choose a module",
        ["Home", "Dataset", "Data Analysis (EDA)", "Prediction",
         "Dashboard", "System Design"],
    )
    st.divider()
    st.caption("Predictions support staff review only. They must not automatically deny or delay a student's service.")

if "request_data" not in st.session_state:
    st.session_state.request_data = None

if page == "Home":
    st.subheader("Project Overview")
    st.write(
        "This application helps authorized student-service staff inspect incoming "
        "request data, explore patterns, and flag requests that may need early human "
        "follow-up. It uses the Decision Tree pipeline from the midterm notebook."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Application modules", "6")
    c2.metric("Model target", "SLA_Breached")
    c3.metric("Decision mode", "Human review")
    st.markdown("### How to use the system")
    st.markdown(
        "1. Open **Dataset** and upload a CSV containing the required intake fields.\n"
        "2. Check the validation summary and explore the data in **Data Analysis (EDA)**.\n"
        "3. Use **Prediction** for one request or a batch of requests.\n"
        "4. Review the flagged queue in **Dashboard**.\n"
        "5. Use **System Design** for the functional requirements, limitations, and workflow."
    )
    st.info(
        "The trained model file is created from your original notebook. Until you export "
        "that fitted pipeline into `models/trained_model.joblib`, the other non-model "
        "pages can still be explored, but predictions will remain unavailable."
    )

elif page == "Dataset":
    st.subheader("Dataset Upload and Validation")
    uploaded = st.file_uploader("Upload student-service request data (CSV)", type=["csv"])
    if uploaded is not None:
        try:
            df = load_and_validate_data(uploaded)
            st.session_state.request_data = df
            st.success(f"Dataset loaded and validated: {len(df):,} rows × {len(df.columns):,} columns.")
        except Exception as exc:
            st.error(f"Dataset could not be accepted: {exc}")
    df = st.session_state.request_data
    if df is not None:
        st.markdown("### Preview")
        st.dataframe(df.head(20), use_container_width=True)
        summary = get_data_quality_summary(df)
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", summary["rows"])
        c2.metric("Columns", summary["columns"])
        c3.metric("Duplicate rows", summary["duplicate_rows"])
        st.markdown("### Missing values by column")
        missing = summary["missing_values"]
        st.dataframe(
            missing.rename("Missing values").to_frame().sort_values("Missing values", ascending=False),
            use_container_width=True
        )
        st.download_button(
            "Download uploaded dataset copy",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="uploaded_student_service_requests.csv",
            mime="text/csv",
        )
    else:
        st.warning("Upload a CSV to see its preview and data-quality summary.")
    st.caption("Required model predictors: " + ", ".join(MODEL_FEATURES))

elif page == "Data Analysis (EDA)":
    df = st.session_state.request_data
    if df is None:
        st.warning("Upload a valid CSV first in the Dataset page.")
    else:
        show_eda(df)

elif page == "Prediction":
    st.subheader("SLA Breach Risk Prediction")
    df = st.session_state.request_data
    try:
        model = load_trained_model()
    except Exception as exc:
        st.error("The trained pipeline is not available yet.")
        st.write(str(exc))
        st.markdown(
            "Open your original midterm notebook, run the cell that fits `decision_tree`, "
            "then run the export snippet included in `SAVE_MODEL_FROM_NOTEBOOK.txt`. "
            "Save the resulting file at `models/trained_model.joblib`."
        )
        st.stop()

    st.write("Enter details known at intake, before processing begins.")
    def options_for(col, fallback):
        if df is not None and col in df.columns:
            vals = df[col].dropna().astype(str).unique().tolist()
            if vals:
                return sorted(vals)
        return fallback

    with st.form("single_request_form"):
        a, b = st.columns(2)
        with a:
            service = st.selectbox("Service Type", options_for("Service_Type", ["Certification", "Records", "ID", "Clearance"]))
            requester = st.selectbox("Requester Type", options_for("Requester_Type", ["Student", "Faculty", "Staff"]))
            weekday = st.selectbox("Submission Weekday", options_for("Submission_Weekday", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]))
            channel = st.selectbox("Requested Channel", options_for("Requested_Channel", ["Portal", "Email", "Walk-in"]))
            priority = st.selectbox("Priority", options_for("Priority", ["Standard", "Urgent"]))
        with b:
            office = st.selectbox("Assigned Office", options_for("Assigned_Office", ["Registrar", "Student Affairs", "Accounting", "IT Office"]))
            queue_default = int(pd.to_numeric(df["Queue_Length"], errors="coerce").median()) if df is not None and "Queue_Length" in df.columns and pd.to_numeric(df["Queue_Length"], errors="coerce").notna().any() else 5
            sla_default = int(pd.to_numeric(df["SLA_Days"], errors="coerce").median()) if df is not None and "SLA_Days" in df.columns and pd.to_numeric(df["SLA_Days"], errors="coerce").notna().any() else 3
            queue = st.number_input("Queue Length at Submission", min_value=0, max_value=100000, value=max(0, queue_default))
            docs = st.selectbox("Required Documents Complete", options_for("Required_Documents_Complete", ["Yes", "No"]))
            sla = st.number_input("SLA Days", min_value=1, max_value=365, value=max(1, sla_default))
        submitted = st.form_submit_button("Predict risk")
    if submitted:
        request = {
            "Service_Type": service,
            "Requester_Type": requester,
            "Submission_Weekday": weekday,
            "Requested_Channel": channel,
            "Priority": priority,
            "Assigned_Office": office,
            "Queue_Length": queue,
            "Required_Documents_Complete": docs,
            "SLA_Days": sla,
        }
        try:
            label, probability = predict_request(model, request)
            risk = "Possible SLA breach — staff review recommended" if int(label) == 1 else "Lower predicted risk — normal monitoring"
            st.markdown("### Prediction result")
            st.warning(risk) if int(label) == 1 else st.success(risk)
            if probability is not None:
                st.metric("Estimated breach probability", f"{probability:.1%}")
            st.write("Input summary")
            st.dataframe(pd.DataFrame([request]), use_container_width=True)
            st.info("This is an estimate, not a guarantee. An authorized staff member must decide what follow-up is appropriate.")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    if df is not None:
        st.markdown("### Batch prediction")
        st.write("Predict all rows in the currently uploaded dataset using the same trained pipeline.")
        if st.button("Predict uploaded dataset"):
            try:
                results = predict_batch(model, df)
                st.session_state.batch_results = results
                st.success(f"Predictions created for {len(results)} rows.")
            except Exception as exc:
                st.error(f"Batch prediction failed: {exc}")
        if st.session_state.get("batch_results") is not None:
            results = st.session_state.batch_results
            st.dataframe(results, use_container_width=True)
            st.download_button(
                "Download batch predictions",
                data=results.to_csv(index=False).encode("utf-8"),
                file_name="sla_risk_predictions.csv",
                mime="text/csv",
            )

elif page == "Dashboard":
    df = st.session_state.request_data
    if df is None:
        st.warning("Upload a valid CSV first in the Dataset page.")
    else:
        try:
            model = load_trained_model()
        except Exception:
            model = None
        show_dashboard(df, model)

elif page == "System Design":
    st.subheader("System Design")
    st.markdown("### Functional requirements")
    st.markdown(
        "1. The system shall allow authorized staff to upload and validate a CSV dataset.\n"
        "2. The system shall show dataset preview, missing values, duplicate rows, and basic descriptive statistics.\n"
        "3. The system shall use the saved preprocessing and Decision Tree pipeline to predict SLA breach risk from intake-time fields.\n"
        "4. The system shall display a review queue and allow staff to download prediction results for follow-up.\n"
        "5. The system shall display model limitations and require human review before administrative action."
    )
    st.markdown("### Non-functional requirements")
    st.markdown(
        "- **Usability:** Navigation and labels should be understandable to student-service staff.\n"
        "- **Reliability:** The app validates required fields and reports prediction/model errors rather than silently continuing.\n"
        "- **Privacy and security:** Only authorized personnel should handle real student data; do not publish identifiable records in a public repository."
    )
    st.markdown("### Analytical workflow")
    st.image("assets/analytical_workflow.png", caption="Intake data → validation and preprocessing → saved model → risk results → authorized human review", use_container_width=True)
    st.markdown("### Use-case diagram")
    st.image("assets/use_case_diagram.png", caption="Main actors and use cases", use_container_width=True)
    st.markdown("### Model limitations")
    st.warning(
        "Predictions depend on the training data and may be wrong. False negatives can miss requests needing help; "
        "false positives can cause unnecessary review. Never automatically reject a request based on this model."
    )
