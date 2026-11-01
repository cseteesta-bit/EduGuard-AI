# 🎓 EduGuard-AI

**Explainable and Uncertainty-Aware Student Academic Risk Prediction** — Master's Advanced AI course project.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cseteesta-bit/EduGuard-AI/blob/main/EduGuard-AI-GitHub-Colab.ipynb)

## Features
- Streamlit GUI: evaluation dashboard, interactive student risk analysis, batch CSV prediction
- Two competing models: Logistic Regression and Random Forest
- Model explanations with SHAP (Random Forest) or linear feature contributions
- Split conformal prediction sets, with empirical coverage, ambiguity, and set-size evaluation
- Group-aware student split into train (50%), validation (20%), calibration (15%), test (15%)
- Early prediction from the first 28 course days, to reduce temporal target leakage
- Reproducible synthetic demonstration mode and real OULAD training mode

## Run in Google Colab
1. Open the **Open in Colab** badge above.
2. Run the notebook cells in order.
3. The notebook clones this repository, installs dependencies, trains a **synthetic demo** model, and launches the Streamlit GUI through Colab's notebook port proxy.
4. To train with OULAD, download and extract the dataset into Google Drive, then run the optional Drive/training notebook cells.

Note: A saved model is **not included** in the repository. The Colab notebook trains it automatically. It will regenerate evaluation files under `models/`.

## Run locally
```bash
git clone https://github.com/cseteesta-bit/EduGuard-AI.git
cd EduGuard-AI
python -m pip install -r requirements.txt
python train.py --demo
streamlit run app.py
```

## Real dataset
Get the **Open University Learning Analytics Dataset (OULAD)** from:
https://archive.ics.uci.edu/dataset/349/open+university+learning+analytics+dataset

Place these four original CSV files in `data/oulad/`:
- studentInfo.csv
- studentVle.csv
- studentAssessment.csv
- assessments.csv

Then execute:
```bash
python train.py --data-dir data/oulad
streamlit run app.py
```
OULAD labels: **Fail/Withdrawn = at risk** and **Pass/Distinction = not at risk**. Features include attempts, course credits, early virtual-learning activity, and early assessments. Only activity in days 0–27 is included; demographics and future outcomes are not used as predictors.

## Experiments
1. Train both models using the training split.
2. Select the model using **validation F1**; never select based on test results.
3. Fit the conformal nonconformity quantile on a separate calibration split.
4. Evaluate untouched test performance: accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, average precision, Brier score, coverage, average prediction-set size, and confusion matrix.
5. Explain individual examples with local feature contributions. These explanations are *not causal*.
6. For academic reporting, use **real OULAD results**, additional random seeds, fairness checks, and a clearly documented outcome definition.

## Project layout
```text
EduGuard-AI/
├── EduGuard-AI-GitHub-Colab.ipynb
├── app.py
├── train.py
├── requirements.txt
├── eduguard/
│   ├── __init__.py
│   ├── data.py
│   └── model.py
├── tests/
│   └── test_core.py
└── README.md
```

## Limitations and safety
Synthetic demonstration results are illustrative and must not be presented as scientific evidence. OULAD is observational; predictive associations do not establish that interventions cause improved results. Conformal coverage is marginal under exchangeability assumptions, not a guarantee for any individual student. This is an educational research prototype, not a production system for automated student decisions. Protect student records and require qualified human review.
