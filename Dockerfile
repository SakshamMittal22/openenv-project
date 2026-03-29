# ──────────────────────────────────────────────────────────────
# AI Email Triage Environment — Hugging Face Spaces compatible
# ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

LABEL maintainer="hackathon-team"
LABEL description="AI Email Triage OpenEnv Environment v2"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY models.py tasks.py reward.py grader.py env.py \
     baseline.py app.py ui.py tests.py openenv.yaml README.md ./

# Optional: copy LLM baseline (won't crash if no API key)
COPY llm_baseline.py ./

# Hugging Face Spaces runs on port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/')" || exit 1

# Launch Gradio UI (judges see visual demo)
CMD ["python", "ui.py"]