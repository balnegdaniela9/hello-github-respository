import pandas as pd

MODEL_FEATURES = [
    "Service_Type",
    "Requester_Type",
    "Submission_Weekday",
    "Requested_Channel",
    "Priority",
    "Assigned_Office",
    "Queue_Length",
    "Required_Documents_Complete",
    "SLA_Days",
]
TARGET_COLUMN = "SLA_Breached"

def load_and_validate_data(uploaded_file):
    """Read and validate a CSV without silently fabricating missing fields."""
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.astype(str).str.strip()
    if df.empty:
        raise ValueError("The uploaded CSV is empty.")
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names were found.")
    missing = [column for column in MODEL_FEATURES if column not in df.columns]
    if missing:
        raise ValueError("Missing required model features: " + ", ".join(missing))
    if "Request_ID" in df.columns and df["Request_ID"].duplicated().any():
        # Duplicate identifiers are reported in the quality summary, not silently removed.
        pass
    return df

def get_data_quality_summary(df):
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": df.isna().sum().astype(int),
    }
