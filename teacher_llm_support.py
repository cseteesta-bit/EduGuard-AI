"""Optional LLM advice for EduGuard-AI teachers.
Never send student identifiers, raw CSVs, or full records to external models.
"""
import os

def risk_summary(probability, candidate_labels, features, contributions=None):
    """Create a minimal, anonymous summary for optional LLM explanation."""
    numeric = ["num_of_prev_attempts", "studied_credits", "early_clicks",
               "active_days", "early_submissions", "early_mean_score"]
    safe = {}
    for key in numeric:
        value = features.get(key)
        try:
            if value is None or value != value:
                safe[key] = None
            else:
                safe[key] = round(float(value), 2)
        except (TypeError, ValueError):
            safe[key] = None
    top = []
    for item in (contributions or [])[:5]:
        label = str(item.get("feature", ""))[:90]
        if not label or any(token in label.lower() for token in ["id_student", "name", "email"]):
            continue
        try:
            value = float(item.get("contribution", 0))
        except (ValueError, TypeError):
            continue
        top.append({"feature": label, "contribution": round(value, 4)})
    return {"risk_probability": round(float(probability), 4),
            "conformal_prediction_set": [v for v in candidate_labels if v in ("At risk", "Not at risk")],
            "features": safe, "top_model_contributions": top}

SYSTEM_GUIDANCE = """You are a teacher-facing academic support assistant.
You receive a short, de-identified model summary, not student records.
Do not invent facts, assessments or diagnoses. Do not claim SHAP attribution is causation.
A risk score is a predictive model output, not an established student outcome.
Conformal sets are marginally calibrated under exchangeability, not individual guarantees.
Explain the probability and uncertainty in plain terms.
Give 3-5 non-punitive, optional support actions for a teacher to consider.
When the set is ambiguous or empty, explicitly recommend human review.
Do not recommend denial of opportunities, discipline, or automatic adverse action.
Mention that recommendations require educator verification and student privacy protections.
Keep response concise, with headings: Interpretation, Suggested Support, Cautions.
"""

def draft_teacher_advice(summary, model_name=None, api_key=None):
    """Use the OpenAI Responses API only when the teacher explicitly requests it."""
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        raise ValueError("OPENAI_API_KEY not configured. LLM suggestions are optional.")
    from openai import OpenAI
    import json
    client = OpenAI(api_key=key, timeout=35, max_retries=1)
    response = client.responses.create(
        model=model_name or os.getenv("EDUGUARD_LLM_MODEL", "gpt-4.1-mini"),
        instructions=SYSTEM_GUIDANCE,
        input="Interpret the following de-identified academic risk summary only:\n" + json.dumps(summary),
        max_output_tokens=450,
        store=False,
    )
    return response.output_text or "The model returned no text."
