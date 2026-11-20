"""Run: uvicorn src.api.main:app --port 7860."""

import logging
from contextlib import asynccontextmanager
from datetime import date
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.schemas import PredictionQuery, PredictionResponse
from src.api.service import PredictionService, ServiceError, Settings

LOGGER = logging.getLogger(__name__)


def create_app(settings=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.service = None
        try:
            app.state.service = PredictionService(settings or Settings.from_env())
        except Exception as exc:
            # Keep health available even when the model/snapshot is not installed yet.
            LOGGER.warning("Prediction service not ready: %s", type(exc).__name__)
        yield

    app = FastAPI(title="California Beach Enterococcus Risk", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(
            status_code=422,
            content={
                "detail": {
                    "code": "invalid_request",
                    "message": "Check beach ID and YYYY-MM-DD date; extra parameters are not allowed",
                }
            },
        )

    @app.exception_handler(ServiceError)
    async def service_error(request, exc):
        return JSONResponse(
            status_code=exc.status,
            content={"detail": {"code": exc.code, "message": exc.message, **exc.context}},
        )

    @app.get("/health")
    def health(request: Request):
        service = request.app.state.service
        if service is None:
            return JSONResponse(
                status_code=503,
                content={"status": "not_ready", "reason": "artifacts_missing_or_incompatible"},
            )
        try:
            return service.health()
        except ServiceError:
            return JSONResponse(
                status_code=503, content={"status": "not_ready", "reason": "no_supported_dates"}
            )

    @app.get("/predict", response_model=PredictionResponse)
    def predict(request: Request, params: Annotated[PredictionQuery, Query()]):
        service = request.app.state.service
        if service is None:
            raise ServiceError("service_not_ready", "Model or history snapshot is unavailable", 503)
        try:
            return service.predict(params.beach, date.fromisoformat(params.date))
        except ServiceError:
            raise
        except Exception:
            LOGGER.exception("Prediction failed")
            raise ServiceError(
                "prediction_failed", "Unable to produce a valid prediction", 503
            ) from None

    return app


app = create_app()
