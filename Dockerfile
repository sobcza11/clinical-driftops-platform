# Dockerfile — Clinical DriftOps reproducible runner
# Local-friendly build (no digest pin). CI can later re-pin a digest.
FROM python:3.11-slim

# Safer defaults
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONPATH=/app

WORKDIR /app

# (Optional) system deps if any wheels need build; otherwise you can remove build-essential
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
  && rm -rf /var/lib/apt/lists/*

# Install deps
COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source & policies (everything, but .dockerignore will prune junk)
COPY . .

# Default entrypoint: your validator CLI
ENTRYPOINT ["python", "-m", "src.api.validate_cli"]
# Default CMD: accept a mounted reports/predictions.csv unless overridden
CMD ["--preds", "reports/predictions.csv"]
