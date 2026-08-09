from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field


class SensitiveInput(BaseModel):
    accessToken: str | None = None
    privateNote: str | None = None


class IncidentInput(BaseModel):
    incidentId: str
    title: str
    service: str
    severity: str
    transcript: str
    allowedRootCauses: list[str]


class ToolDefinition(BaseModel):
    name: str
    description: str
    inputSchema: dict[str, Any]


class PolicyInput(BaseModel):
    maximumDiagnostics: int = Field(ge=1, le=3)
    effectTools: list[str]
    approvalRequiredFor: list[str]
    doNotExport: list[str]


class CreateIncidentRequest(BaseModel):
    profile: str
    runId: str = Field(min_length=1)
    agentName: str = Field(min_length=1)
    publicMarker: str = Field(min_length=1)
    sensitive: SensitiveInput
    incident: IncidentInput
    toolCatalog: list[ToolDefinition]
    policy: PolicyInput


class Outcome(BaseModel):
    actionId: str
    callId: str
    attempt: int
    status: int
    errorType: str | None = None
    resultClass: str | None = None
    nonce: str | None = None


class ApprovalDecision(BaseModel):
    approvalId: str
    decision: str
    nonce: str


class ReceiptRequest(BaseModel):
    receiptId: str
    outcomes: list[Outcome] = Field(default_factory=list)
    approvals: list[ApprovalDecision] = Field(default_factory=list)


class PlannerOutput(BaseModel):
    rootCause: str
    evidence: list[str]
    diagnostics: list[dict[str, Any]]
    effectPlan: dict[str, Any]


TRACEPARENT_RE = re.compile(
    r"^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$"
)

SPAN_KIND_INTERNAL = 1
SPAN_KIND_SERVER = 2
SPAN_KIND_CLIENT = 3
