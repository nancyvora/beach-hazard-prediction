FROM python:3.13.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFLOW_DISABLE_TELEMETRY=true \
    MLFLOW_DISABLE_AGENT_HINT=1

WORKDIR /home/user/app
COPY requirements-serve.txt ./
RUN pip install -r requirements-serve.txt && useradd --create-home --uid 1000 user
COPY --chown=user:user src ./src
COPY --chown=user:user deploy ./deploy
USER user

ENV MODEL_URI=/home/user/app/artifacts/model \
    FEATURE_REPORT_PATH=/home/user/app/artifacts/feature_report.json \
    STATION_CATALOG_PATH=/home/user/app/artifacts/station_catalog.parquet \
    HISTORY_PATH=/home/user/app/artifacts/station_history.parquet \
    HISTORY_METADATA_PATH=/home/user/app/artifacts/history_metadata.json

EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
