from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from eduguard.data import FEATURES
from eduguard.model import load_bundle, infer, explanations

st.set_page_config(page_title="EduGuard-AI", page_icon="🎓", layout="wide")
st.title("🎓 EduGuard-AI")
st.caption("Explainable, uncertainty-aware student academic risk prediction | Master's Advanced AI")
folder = Path("models")
if not (folder / "eduguard.joblib").exists():
    st.warning("No trained model. Run: python train.py --demo (or python train.py --data-dir data/oulad)")
    st.stop()

@st.cache_resource
def get_model():
    return load_bundle("models")

bundle = get_model()
stats = json.loads((folder / "metrics.json").read_text())
if bundle["demo"]:
    st.error("SYNTHETIC DEMONSTRATION ONLY — not evidence of real-student prediction quality.")
else:
    st.info("OULAD research prototype — do not use for automated individual educational decisions.")

page = st.sidebar.radio("Navigate", ["Overview & Evaluation", "Predict Student Risk", "Batch Predictions", "About & Limitations"])
if page == "Overview & Evaluation":
    st.subheader("Held-out test evaluation")
    st.write("Selected model:", stats["selected_model"])
    cols = st.columns(4)
    for col, (key, n) in zip(cols, stats["sizes"].items()):
        col.metric(key.title(), n)
    frame = pd.DataFrame([{"Model": name, **{k:v for k,v in m["test"].items() if isinstance(v,(float,int))}} for name,m in stats["models"].items()])
    st.dataframe(frame.set_index("Model").style.format(precision=3), use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        subset=frame[["Model","f1","roc_auc"]].melt(id_vars="Model", var_name="Metric")
        st.plotly_chart(px.bar(subset,x="Model",y="value",color="Metric",barmode="group",title="Model comparison"),use_container_width=True)
    with c2:
        confusion=np.asarray(stats["models"][stats["selected_model"]]["test"]["confusion_matrix"])
        st.plotly_chart(px.imshow(confusion,text_auto=True,x=["Not at risk","At risk"],y=["Not at risk","At risk"],title="Test confusion matrix"),use_container_width=True)
    st.caption("Conformal coverage is marginal under exchangeability, not a guarantee for each student.")
elif page == "Predict Student Risk":
    st.subheader("Individual risk prediction")
    examples=pd.read_csv(folder/"examples.csv")
    idx=st.selectbox("Example student",range(len(examples)),format_func=lambda i:f"Example {i+1}")
    source=examples.iloc[idx]
    with st.form("student"):
        a,b=st.columns(2)
        with a:
            module=st.text_input("Course module",str(source["code_module"]))
            presentation=st.text_input("Presentation",str(source["code_presentation"]))
            prev=st.number_input("Previous attempts",0,20,int(source["num_of_prev_attempts"]))
            credits=st.number_input("Studied credits",0,500,int(source["studied_credits"]))
        with b:
            clicks=st.number_input("VLE clicks (first 28 days)",0,100000,int(source["early_clicks"]))
            days=st.number_input("Active days (first 28 days)",0,28,int(source["active_days"]))
            submissions=st.number_input("Submitted assessments",0,50,int(source["early_submissions"]))
            current_score=0 if pd.isna(source["early_mean_score"]) else float(source["early_mean_score"])
            score=st.number_input("Mean early score",0.,100.,current_score)
        clicked=st.form_submit_button("Analyze risk",type="primary")
    if clicked:
        row=pd.DataFrame([{"code_module":module,"code_presentation":presentation,"num_of_prev_attempts":prev,
          "studied_credits":credits,"early_clicks":clicks,"active_days":days,"early_submissions":submissions,
          "early_mean_score":score if submissions else np.nan}])
        probabilities,sets=infer(bundle,row)
        probability=float(probabilities[0])
        options=[name for name,flag in zip(["Not at risk","At risk"],sets[0]) if flag]
        a,b=st.columns(2)
        a.metric("Estimated risk",f"{probability:.1%}")
        b.metric("Conformal prediction set",", ".join(options) if options else "Empty")
        if len(options)!=1:st.warning("Uncertain prediction — human review is recommended.")
        st.progress(probability)
        try:
            exp=explanations(bundle,row).head(12)
            st.plotly_chart(px.bar(exp.sort_values("contribution"),x="contribution",y="feature",orientation="h",
                title="Local SHAP / linear contributions"),use_container_width=True)
        except Exception as error:
            st.warning(f"Explanation unavailable: {error}")
        st.caption("Attributions are descriptive, not causal effects.")
        recommendations=[]
        if days<10:recommendations.append("Encourage consistent engagement with learning materials.")
        if not submissions:recommendations.append("Check assessment deadlines and offer submission support.")
        if submissions and score<50:recommendations.append("Offer targeted feedback on early assessments.")
        if prev>0:recommendations.append("Offer academic adviser consultation.")
        st.subheader("Suggested human-reviewed support")
        for r in recommendations or ["Maintain current progress and monitor engagement."]:
            st.write("• "+r)
elif page == "Batch Predictions":
    st.subheader("Upload student CSV")
    template=pd.read_csv(folder/"examples.csv")[FEATURES].head(10)
    st.download_button("Download CSV template",template.to_csv(index=False),"eduguard_input_template.csv","text/csv")
    upload=st.file_uploader("CSV input",type=["csv"])
    if upload:
        frame=pd.read_csv(upload)
        missing=[c for c in FEATURES if c not in frame.columns]
        if missing:st.error("Missing: "+", ".join(missing))
        elif len(frame)>20000:st.error("Maximum 20,000 records per upload")
        else:
            prob,sets=infer(bundle,frame)
            out=frame.copy()
            out["predicted_risk_probability"]=prob
            out["prediction_set"]=[", ".join([v for v,yes in zip(["Not at risk","At risk"],s) if yes]) or "Empty" for s in sets]
            st.dataframe(out.head(100),use_container_width=True)
            st.download_button("Download predictions",out.to_csv(index=False),"eduguard_predictions.csv","text/csv")
else:
    st.markdown("""### Task and methodology
Final Fail/Withdrawn outcomes are considered academic risk; Pass/Distinction are not. Only days 0–27 are used.

Models: Logistic Regression and Random Forest. Select using validation F1, then calibrate conformal prediction sets on an independent calibration split and measure performance on untouched test data.

### Limitations
Synthetic demo results are not research findings. The real OULAD dataset is observational, so associations do not prove causal effects. Review fairness, missingness, distribution shift, privacy, and policy implications before real-world deployment. Students should not be subject to fully automated interventions.
""")
