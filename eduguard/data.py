from pathlib import Path
import numpy as np
import pandas as pd

KEYS = ["code_module", "code_presentation", "id_student"]
NUM = ["num_of_prev_attempts", "studied_credits", "early_clicks", "active_days", "early_submissions", "early_mean_score"]
CAT = ["code_module", "code_presentation"]
FEATURES = NUM + CAT

def load_oulad(folder, cutoff=28):
    """Aggregate OULAD observations from course days 0 through 27 only."""
    folder = Path(folder)
    required = ["studentInfo.csv", "studentVle.csv", "studentAssessment.csv", "assessments.csv"]
    for name in required:
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Missing {folder / name}")
    info = pd.read_csv(folder / "studentInfo.csv")
    info = info[info.final_result.isin(["Pass", "Distinction", "Fail", "Withdrawn"])].copy()
    info["at_risk"] = info.final_result.isin(["Fail", "Withdrawn"]).astype(int)
    info = info[KEYS + ["num_of_prev_attempts", "studied_credits", "at_risk"]]
    clicks, days = {}, {}
    for chunk in pd.read_csv(folder / "studentVle.csv", usecols=KEYS + ["date", "sum_click"], chunksize=350000):
        chunk["date"] = pd.to_numeric(chunk["date"], errors="coerce")
        chunk["sum_click"] = pd.to_numeric(chunk["sum_click"], errors="coerce").fillna(0)
        chunk = chunk[(chunk.date >= 0) & (chunk.date < cutoff)]
        for key, part in chunk.groupby(KEYS, sort=False):
            clicks[key] = clicks.get(key, 0) + float(part.sum_click.sum())
            days.setdefault(key, set()).update(part.date.unique().tolist())
    vle = pd.DataFrame([(*key, count, len(days[key])) for key, count in clicks.items()], columns=KEYS + ["early_clicks", "active_days"])
    meta = pd.read_csv(folder / "assessments.csv", usecols=["id_assessment", "code_module", "code_presentation", "date", "assessment_type"])
    sa = pd.read_csv(folder / "studentAssessment.csv", usecols=["id_assessment", "id_student", "date_submitted", "score"])
    # Some OULAD assessment dates are missing or stored as strings; coercion prevents
    # string-vs-integer comparison errors and excludes unknown due dates.
    meta["date"] = pd.to_numeric(meta["date"], errors="coerce")
    sa["date_submitted"] = pd.to_numeric(sa["date_submitted"], errors="coerce")
    sa["score"] = pd.to_numeric(sa["score"], errors="coerce")
    sa = sa.merge(meta, on="id_assessment", how="inner", validate="many_to_one")
    sa = sa[(sa.date_submitted >= 0) & (sa.date_submitted < cutoff) & sa.date.notna() & (sa.date < cutoff) & (sa.assessment_type != "Exam")]
    assessments = sa.groupby(KEYS, as_index=False).agg(early_submissions=("score", "size"), early_mean_score=("score", "mean"))
    frame = info.merge(vle, on=KEYS, how="left").merge(assessments, on=KEYS, how="left")
    for name in ["early_clicks", "active_days", "early_submissions"]:
        frame[name] = frame[name].fillna(0)
    return frame

def synthetic_data(n=1400, seed=42):
    """Synthetic illustration; NEVER use synthetic metrics as scientific findings."""
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        prev = int(r.choice([0, 1, 2, 3], p=[.76, .16, .06, .02]))
        credits = int(r.choice([30, 60, 90, 120]))
        active = int(r.binomial(28, r.uniform(.1, .9)))
        clicks = int(r.poisson(4 + active * 9))
        submits = int(r.binomial(3, .15 + .7 * active / 28))
        score = float(np.clip(r.normal(54 + active * .8 - 5 * prev, 18), 0, 100)) if submits else np.nan
        linear = 1.2 - .075 * active - .011 * clicks + .65 * prev + .004 * (credits - 60) - (.035 * (score - 55) if submits else 0) + (.5 if not submits else 0)
        risk = int(r.random() < 1 / (1 + np.exp(-linear)))
        rows.append((r.choice(["AAA", "BBB", "CCC"]), r.choice(["2013J", "2014B", "2014J"]), 100000 + i, prev, credits, clicks, active, submits, score, risk))
    return pd.DataFrame(rows, columns=KEYS + NUM + ["at_risk"])
