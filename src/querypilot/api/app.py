from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from querypilot.api.schemas import (
    HistoryResponse,
    RunRequest,
    RunResponse,
    StatusResponse,
)
from querypilot.services.agent_service import (
    AgentService,
    InvalidRunStateError,
    RunNotFoundError,
)


def _run_response(record: dict[str, Any]) -> RunResponse:
    return RunResponse(run_id=record["id"], **_pick(record, RunResponse, skip={"run_id"}))


def _pick(record: dict[str, Any], model: type[BaseModel], skip: set[str]) -> dict[str, Any]:
    return {k: record[k] for k in model.model_fields if k not in skip and k in record}


def create_app(service: AgentService | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
        if app.state.service is None:
            from querypilot.bootstrap import build_default_service

            app.state.service = build_default_service()
        yield

    app = FastAPI(title="QueryPilot", version="0.1.0", lifespan=lifespan)
    app.state.service = service

    @app.exception_handler(RunNotFoundError)
    async def _not_found(_: Request, exc: RunNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": f"Run not found: {exc}"})

    @app.exception_handler(InvalidRunStateError)
    async def _conflict(_: Request, exc: InvalidRunStateError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    # Endpoints are sync on purpose: graph.invoke blocks, so FastAPI runs them in a thread pool.
    @app.post("/agent/run", response_model=RunResponse)
    def run(body: RunRequest, request: Request) -> RunResponse:
        return _run_response(request.app.state.service.run(body.question))

    @app.get("/agent/status/{run_id}", response_model=StatusResponse)
    def status(run_id: str, request: Request) -> StatusResponse:
        record = request.app.state.service.get(run_id)
        return StatusResponse(
            run_id=run_id, **_pick(record, StatusResponse, skip={"run_id"})
        )

    @app.post("/agent/approve/{run_id}", response_model=RunResponse)
    def approve(run_id: str, request: Request) -> RunResponse:
        return _run_response(request.app.state.service.approve(run_id))

    @app.post("/agent/reject/{run_id}", response_model=RunResponse)
    def reject(run_id: str, request: Request) -> RunResponse:
        return _run_response(request.app.state.service.reject(run_id))

    @app.get("/agent/history/{run_id}", response_model=HistoryResponse)
    def history(run_id: str, request: Request) -> HistoryResponse:
        record = request.app.state.service.get(run_id)
        return HistoryResponse(
            run_id=run_id, **_pick(record, HistoryResponse, skip={"run_id"})
        )

    return app
