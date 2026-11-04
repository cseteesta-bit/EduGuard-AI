# EduGuard-AI — Teacher Login + Optional LLM Extension

**This is separate from the original release.** The existing `app.py`, `train.py`, `requirements.txt`, and `EduGuard-AI-GitHub-Colab.ipynb` remain unchanged.

## Open in Colab
[Open the separate teacher + LLM notebook](https://colab.research.google.com/github/cseteesta-bit/EduGuard-AI/blob/main/EduGuard-AI-Teacher-LLM-Colab.ipynb)

## Features
- Separate Streamlit teacher portal, protected by a temporary session login.
- Teacher credentials from Colab Secrets, never hardcoded or committed.
- Same OULAD preprocessing and ML training as the original project.
- Existing test metrics, risk scoring, explainability and conformal prediction.
- Rule-based academic support.
- Optional LLM teacher guidance **only on click and after explicit consent**.
- Batch prediction without LLM calls.

## Setup
1. Open the separate notebook in Colab, disconnect any old runtime, and use **Runtime → Run all**.
2. In Colab **Secrets** (key icon), set `EDUGUARD_TEACHER_USER` and `EDUGUARD_TEACHER_PASSWORD`. Password must be at least 12 characters. Enable notebook access.
3. For optional LLM explanations, set `OPENAI_API_KEY` as a separate secret. It is **not** included in ChatGPT subscriptions and can incur API charges.
4. Optionally set `EDUGUARD_LLM_MODEL` to a model available to your API account (default `gpt-4.1-mini`).
5. The notebook installs dependencies, downloads OULAD, trains the original model, and opens the teacher portal on port 8502.
6. Log in with your secret credentials. Open Student Risk + LLM to generate optional AI-written teacher guidance.

## Files added
- `EduGuard-AI-Teacher-LLM-Colab.ipynb`: new notebook
- `app_teacher_llm.py`: separate Streamlit application
- `teacher_llm_support.py`: limited-summary LLM caller
- `requirements-teacher-llm.txt`: extra dependency list
- `TEACHER_LLM_README.md`: this guide

## Security / privacy limits
This is a **course demo only**. The Colab proxy requires disabling Streamlit origin/CSRF checks, which is unsuitable for public deployment. Never use real student identifiers or confidential university records in this demo. The login is a single teacher password, not a production role-based system. For actual use, deploy with HTTPS, secure authentication, access control, audit logs, session management, rate limiting, secure storage, and privacy review.

The optional external LLM receives a **de-identified numerical summary** only when a teacher enables consent and requests an explanation. Input can still contain potentially sensitive educational indicators, so do not send actual students' confidential information without legal and institutional approval. LLM text can contain mistakes and does not determine student academic outcomes.

The ML model's risk score and conformal set do **not** change when LLM guidance is enabled. No claims of performance improvement are made without experiments.
