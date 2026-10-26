# Student Service Request Monitoring System

A Streamlit decision-support application for exploring student-service request data and flagging possible SLA breach risk for authorized human review.

## Pages
- Home
- Dataset upload and validation
- Data Analysis (EDA)
- Prediction (single request and batch)
- Dashboard
- System Design

## Important: export the actual midterm model
The trained model artifact is not fabricated or replaced with a newly invented model. Export the fitted `decision_tree` pipeline from your original midterm notebook using `SAVE_MODEL_FROM_NOTEBOOK.txt`. Put the resulting file at:

`models/trained_model.joblib`

This pipeline should be fitted using the same notebook and predictors:
`Service_Type`, `Requester_Type`, `Submission_Weekday`, `Requested_Channel`, `Priority`, `Assigned_Office`, `Queue_Length`, `Required_Documents_Complete`, `SLA_Days`.

The target is `SLA_Breached` (1 = breached; 0 = not breached). Do not use `Processing_Days`, `Status`, `Reopened`, `Satisfaction_Score`, or `Request_ID` as predictors for intake-time prediction.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in this project folder.
3. Install packages: `python -m pip install -r requirements.txt`
4. Run: `streamlit run app.py`
5. Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Data
Upload a CSV with all nine required model predictor columns. Historical `SLA_Breached` is optional for charts; predictions only use the nine intake-time predictors.

## Ethical use
This app is decision support only. A prediction is not a guaranteed outcome and must not automatically deny a student's service. Authorized staff decide whether follow-up is needed.

## Before deployment
- Export the real model pipeline.
- Test the app with the notebook's eligible data and several new intake cases.
- Confirm that predictions and metrics align with the notebook.
- Never commit real student personal information or credentials to a public repository.
