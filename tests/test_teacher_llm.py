from teacher_llm_support import risk_summary, SYSTEM_GUIDANCE

def test_llm_summary_omits_identifiers():
    features = {
        "id_student": 123456, "student_name": "Private Name",
        "code_module": "AAA", "code_presentation": "2014J",
        "num_of_prev_attempts": 1, "studied_credits": 60,
        "early_clicks": 120, "active_days": 14,
        "early_submissions": 2, "early_mean_score": 69.5,
    }
    result = risk_summary(.71, ["At risk"], features, [
        {"feature": "num__early_clicks", "contribution": .05},
        {"feature": "student_name", "contribution": .5},
    ])
    serialized = str(result)
    assert "Private Name" not in serialized
    assert "123456" not in serialized
    assert "2014J" not in serialized
    assert "AAA" not in serialized
    assert result["features"]["active_days"] == 14
    assert result["conformal_prediction_set"] == ["At risk"]
    assert len(result["top_model_contributions"]) == 1

def test_no_identifiers_in_llm_guidance():
    assert "Do not invent" in SYSTEM_GUIDANCE
