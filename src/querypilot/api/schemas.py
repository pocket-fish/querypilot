from typing import Any

from pydantic import BaseModel, Field


class RunRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class PendingApproval(BaseModel):
    type: str
    tool: str
    sql: str | None
    message: str


class RunResponse(BaseModel):
    run_id: str
    status: str
    final_answer: str | None = None
    pending_approval: PendingApproval | None = None
    approval_expires_at: str | None = None
    error: str | None = None


class StatusResponse(RunResponse):
    current_step: str | None = None
    iterations: int = 0
    updated_at: str


class HistoryResponse(BaseModel):
    run_id: str
    question: str
    status: str
    visited_nodes: list[str]
    tools_used: list[str]
    sql: str | None = None
    tool_result: dict[str, Any] | None = None
    iterations: int
    final_answer: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str
