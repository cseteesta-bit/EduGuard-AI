"""Teacher-only research demonstration with optional LLM support.
Separate from the original EduGuard-AI Streamlit interface.
"""
import os
import hmac
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from eduguard.data import FEATURES
from eduguard.model import load_bundle, infer, explanations
from teacher_llm_support import risk_summary, draft_teacher_advice

st.set_page_config(page_title="EduGuard-AI | Teacher Portal", page_icon="🎓", layout="wide")
st.title("🎓 EduGuard-AI | Teacher Portal")
st.caption("Optional LLM assistance · Explainable predictions · Human-reviewed academic support")

MAX_FAILURES = 5
SESSION_SECONDS = 45 * 60

def auth_gate():
    username = os.getenv("EDUGUARD_TEACHER_USER", "")
    password = os.getenv("EDUGUARD_TEACHER_PASSWORD", "")
    if not username or not password or len(password) < 12:
        st.error("Teacher credentials missing. Add EDUGUARD_TEACHER_USER and a strong EDUGUARD_TEACHER_PASSWORD (12+ characters) to Colab Secrets.")
        st.stop()
    if st.session_state.get("teacher_authenticated"):
        if time.monotonic() - st.session_state.get("login_time", 0) < SESSION_SECONDS:
            return
        st.session_state.pop("teacher_authenticated", None)
        st.warning("Your session expired. Sign in again.")
    with st.form("login_form"):
        st.subheader("Teacher sign-in")
        user = st.text_input("Teacher username", max_chars=128)
        secret = st.text_input("Password", type="password", max_chars=200)
        login = st.form_submit_button("Sign in", type="primary")
    if login:
        failures = st.session_state.get("login_failures", 0)
        if failures >= MAX_FAILURES:
            st.error("Too many unsuccessful sign-in attempts for this session. Restart the browser session.")
        elif hmac.compare_digest(user.encode(), username.encode()) and hmac.compare_digest(secret.encode(), password.encode()):
            st.session_state["teacher_authenticated"] = True
            st.session_state["login_time"] = time.monotonic()
            st.session_state["login_failures"] = 0
            st.rerun()
        else:
            st.session_state["login_failures"] = failures + 1
            st.error("Invalid credentials.")
    st.caption("Classroom demo only. Session login is not suitable as a production identity system.")
    st.stop()

auth_gate()

with st.sidebar:
    st.success("Teacher session active")
    if st.button("Log out"):
        for field in ["teacher_authenticated", "login_time", "last_risk", "last_features", "last_top_contributions", "llm_answer"]:
            st.session_state.pop(field, None)
        st.rerun()
    menu = st.radio("Navigation", ["Dashboard & Evaluation", "Student Risk + LLM", "Batch CSV Prediction", "Responsible Use"])

model_dir = Path("models")
if not (model_dir / "eduguard.joblib").is_file():
    st.error("No model found. Run python train.py --data-dir data/oulad before launching.")
    st.stop()

@st.cache_resource
def get_bundle():
    return load_bundle("models")

bundle = get_bundle()
import json
metrics_path = model_dir / "metrics.json"
if not metrics_path.exists():
    st.error("Evaluation metrics missing. Train the model using train.py.")
    st.stop()
metrics = json.loads(metrics_path.read_text())
st.info("Research prototype. Human evaluation required; never use as an automatic student penalty or admission decision.")
if bundle.get("demo"):
    st.warning("Synthetic demo model in use. These metrics are not valid real-world findings.")

if menu == "Dashboard & Evaluation":
    st.subheader("Test-set model comparison")
    rows = []
    for name, details in metrics["models"].items():
        m = details["test"]
        rows.append({"Model": name, "Accuracy": m.get("accuracy"),
                     "F1-score": m.get("f1"), "ROC-AUC": m.get("roc_auc"),
                     "Coverage": m.get("coverage"), "Mean set size": m.get("avg_set_size")})
    data = pd.DataFrame(rows)
    st.dataframe(data, hide_index=True, use_container_width=True)
    st.write("Selected model:", metrics["selected_model"])
    tidy = data.melt(id_vars="Model", value_vars=["Accuracy", "F1-score", "ROC-AUC", "Coverage"],
                     var_name="Metric", value_name="Value")
    st.plotly_chart(px.bar(tidy, x="Metric", y="Value", color="Model", barmode="group", title="Held-out test comparison"),
                    use_container_width=True)
    cm = metrics["models"][metrics["selected_model"]]["test"].get("confusion_matrix")
    if cm is not None:
        st.plotly_chart(px.imshow(np.asarray(cm), x=["Not at risk","At risk"],
                                 y=["Not at risk", "At risk"], text_auto=True,
                                 title="Selected model confusion matrix"), use_container_width=True)
    st.caption("Model selection uses validation F1; results shown here are from held-out test data.")

elif menu == "Student Risk + LLM":
    st.subheader("Teacher-assisted individual assessment")
    examples_path = model_dir / "examples.csv"
    if not examples_path.exists():
        st.error("Example records missing. Re-run training.")
        st.stop()
    examples = pd.read_csv(examples_path)
    idx = st.selectbox("Select a sample student record (not sent to LLM)", range(len(examples)),
                       format_func=lambda i: f"Sample record {i+1}")
    source = examples.iloc[int(idx)]
    with st.form("risk_form"):
        left, right = st.columns(2)
        with left:
            module = st.text_input("Course module", value=str(source["code_module"]))
            presentation = st.text_input("Course presentation", value=str(source["code_presentation"]))
            prev = st.number_input("Previous attempts", min_value=0, max_value=20, value=int(source["num_of_prev_attempts"]))
            credits = st.number_input("Studied credits", min_value=0, max_value=500, value=int(source["studied_credits"]))
        with right:
            clicks = st.number_input("VLE clicks (days 0–27)", min_value=0, max_value=100000, value=int(source["early_clicks"]))
            active = st.number_input("Active learning days", min_value=0, max_value=28, value=int(source["active_days"]))
            submissions = st.number_input("Early submissions", min_value=0, max_value=50, value=int(source["early_submissions"]))
            saved_score = 0.0 if pd.isna(source["early_mean_score"]) else float(source["early_mean_score"])
            score = st.number_input("Early mean assessment score", min_value=0.0, max_value=100.0,
                                    value=float(saved_score), step=1.0)
        run = st.form_submit_button("Analyze student risk", type="primary")
    if run:
        row = pd.DataFrame([{"code_module": module, "code_presentation": presentation,
                             "num_of_prev_attempts": prev, "studied_credits": credits,
                             "early_clicks": clicks, "active_days": active,
                             "early_submissions": submissions,
                             "early_mean_score": score if submissions else np.nan}])
        probs, sets = infer(bundle, row)
        prob = float(probs[0])
        names = [k for k, yes in zip(["Not at risk", "At risk"], sets[0]) if yes]
        st.session_state["last_risk"] = {"probability": prob, "set": names}
        st.session_state["last_features"] = row.iloc[0].to_dict()
        st.session_state["last_top_contributions"] = []
        st.session_state.pop("llm_answer", None)
        try:
            contributions = explanations(bundle, row).head(12)
            st.session_state["last_top_contributions"] = contributions[["feature","contribution"]].to_dict("records")
            st.plotly_chart(px.bar(contributions.sort_values("contribution"),
                                   x="contribution", y="feature", orientation="h",
                                   title="Model feature contributions"), use_container_width=True)
        except Exception as exc:
            st.warning("Explanation could not be displayed: " + str(exc))
    if "last_risk" in st.session_state:
        data = st.session_state["last_risk"]
        p = data["probability"]
        labels = data["set"]
        a, b = st.columns(2)
        a.metric("Estimated academic risk", f"{p:.1%}")
        b.metric("Conformal prediction set", ", ".join(labels) if labels else "Empty set")
        if len(labels) != 1:
            st.warning("Prediction is ambiguous or empty; human review is especially important.")
        st.caption("The score predicts a historical class, not a definite future outcome.")
        features = st.session_state["last_features"]
        suggestions = []
        if features["active_days"] < 10:
            suggestions.append("Review patterns of weekly learning engagement.")
        if features["early_submissions"] == 0:
            suggestions.append("Check deadlines and offer assistance with assessments.")
        if features["early_submissions"] > 0 and features["early_mean_score"] < 50:
            suggestions.append("Offer feedback on early assessment concepts.")
        if features["num_of_prev_attempts"] > 0:
            suggestions.append("Consider an optional academic-adviser consultation.")
        st.markdown("#### Rule-based support suggestions")
        for item in suggestions or ["Continue normal check-ins and monitor progress."]:
            st.write("• " + item)
        st.divider()
        st.markdown("#### Optional LLM teacher assistant")
        st.caption("An external LLM is contacted only when you enable consent and press Generate. "
                   "No student ID, name, course identifier, or CSV file is transmitted.")
        configured = bool(os.getenv("OPENAI_API_KEY"))
        if not configured:
            st.info("LLM not configured: add OPENAI_API_KEY to Colab Secrets to enable this optional feature. Prediction still works.")
        consent = st.checkbox("I understand that a de-identified numeric prediction summary will be sent to an external API.")
        if st.button("Generate LLM support explanation", disabled=not (consent and configured)):
            with st.spinner("Generating teacher-facing support guidance..."):
                safe_summary = risk_summary(p, labels, features,
                                            st.session_state.get("last_top_contributions", []))
                try:
                    answer = draft_teacher_advice(safe_summary)
                    st.session_state["llm_answer"] = answer
                except Exception as exc:
                    st.error("LLM request failed; original model prediction is unaffected. " + str(exc))
        if st.session_state.get("llm_answer"):
            st.markdown(st.session_state["llm_answer"])
            st.caption("AI-generated suggestions must be checked by an educator.")

elif menu == "Batch CSV Prediction":
    st.subheader("Batch prediction — teacher use")
    template = pd.read_csv(model_dir / "examples.csv")[FEATURES].head(10)
    st.download_button("Download feature-only CSV template", template.to_csv(index=False),
                       "teacher_batch_template.csv", "text/csv")
    upload = st.file_uploader("Upload CSV with the required feature columns", type=["csv"])
    if upload:
        incoming = pd.read_csv(upload)
        missing = [name for name in FEATURES if name not in incoming.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        elif len(incoming) > 10000:
            st.error("Batch limit is 10,000 records.")
        else:
            # Export only known model features and predictions, not incidental identifiers.
            clean = incoming[FEATURES].copy()
            probs, sets = infer(bundle, clean)
            clean["risk_probability"] = probs
            clean["prediction_set"] = [", ".join(
                name for name, ok in zip(["Not at risk","At risk"], labels) if ok) or "Empty"
                                        for labels in sets]
            st.dataframe(clean.head(100), use_container_width=True, hide_index=True)
            st.download_button("Download results", clean.to_csv(index=False),
                               "teacher_batch_predictions.csv", "text/csv")
            st.caption("Batch rows are processed locally by the model; no LLM call is made.")

else:
    st.markdown("""### Responsible use
- This is a **single-teacher classroom prototype**, not institutional authentication.
- Teacher credentials and API keys are set in Colab Secrets and never committed to GitHub.
- The optional API receives only aggregate numeric features, prediction and a few model contributions.
- LLM recommendations can be inaccurate; they must not determine grades, discipline, admissions or student access.
- The system has not been audited for role management, CSRF, rate limits, persistent session management or FERPA/GDPR compliance.
- The dataset is observational; explanations are **not causal**, and conformal coverage is marginal under exchangeability.
- Colab hosting is temporary. Do not host real students' confidential information on a public proxy URL.
""")
