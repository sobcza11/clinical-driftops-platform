# Clinical DriftOps — Phase IV container
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy code and data
COPY src ./src
COPY data ./data
COPY reports ./reports

# Default command: build the Evidently drift report
CMD ["python", "-m", "src.make_small_drift_report"]