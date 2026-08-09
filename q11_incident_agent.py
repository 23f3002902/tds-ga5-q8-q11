from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException, Request

from q11_models import (
    CreateIncidentRequest,
    Outcome,
    ReceiptRequest,
)
from q11_planner import (
    AIProviderError,
    PlannerOutputError,
    call_planner,
)
from q11_state_machine import (
    build_initial_state,
    process_approval,
    process_effect_outcome,
    process_outcome,
    rebuild_otlp,
)
from q11_store import init_db, load_receipt, load_run, save_receipt, save_run
from q11_utils import (
    opaque_id,
    parse_traceparent,
    new_trace_id,
    new_span_id,
    sha256_hex,
)

router = APIRouter()


@router.on_event("startup")
def startup() -> None:
    init_db()


def _planning_failed_state(request_hash: str) -> dict:
    return {
        "stage": "planning_failed",
        "requestHash": request_hash,
        "currentResponse": {
            "status": "failed",
            "detail": "AI planning failed after exhausting all providers",
        },
    }


@router.post("/v2/incidents")
async def create_incident(
    body: CreateIncidentRequest,
    request: Request,
    traceparent: str | None = Header(default=None),
    tracestate: str | None = Header(default=None),
):
    if body.profile != "ga5-incident-agent/v2":
        raise HTTPException(status_code=422, detail="Unsupported profile")

    raw_body = await request.json()
    request_hash = sha256_hex(raw_body)
    existing = load_run(body.runId)

    if existing:
        if existing["request_hash"] == request_hash:
            state = existing["state"]
            if state.get("stage") == "planning_failed":
                raise HTTPException(status_code=503, detail="AI planning failed after exhausting all providers")
            return state["currentResponse"]
        raise HTTPException(
            status_code=409,
            detail="runId already exists with different content",
        )

    incoming_trace_id: str
    incoming_span_id: str | None
    if traceparent:
        try:
            incoming_trace_id, incoming_span_id_str, _ = parse_traceparent(traceparent)
            incoming_span_id = incoming_span_id_str
        except ValueError:
            raise HTTPException(status_code=422, detail="Invalid traceparent")
    else:
        incoming_trace_id = new_trace_id()
        incoming_span_id = None

    fail_state = _planning_failed_state(request_hash)
    save_run(body.runId, request_hash, fail_state)

    try:
        plan, model_name = await call_planner(body)
    except AIProviderError:
        raise HTTPException(status_code=503, detail="AI planning failed after exhausting all providers")
    except PlannerOutputError:
        raise HTTPException(status_code=503, detail="AI planning failed after exhausting all providers")

    state = build_initial_state(
        request=body,
        request_hash=request_hash,
        plan=plan,
        incoming_trace_id=incoming_trace_id,
        incoming_span_id=incoming_span_id,
        incoming_tracestate=tracestate,
        model_name=model_name,
    )

    state["otlp"] = rebuild_otlp(state)
    state["currentResponse"]["otlp"] = state["otlp"]

    _redact_check(state["currentResponse"], body)

    save_run(body.runId, request_hash, state)

    return state["currentResponse"]


@router.post("/v2/incidents/{run_id}/receipts")
async def submit_receipts(
    run_id: str,
    body: ReceiptRequest,
    request: Request,
):
    raw_body = await request.json()
    receipt_hash = sha256_hex(raw_body)
    existing_receipt = load_receipt(run_id, body.receiptId)

    if existing_receipt:
        if existing_receipt["receipt_hash"] == receipt_hash:
            return existing_receipt["response"]
        raise HTTPException(
            status_code=409,
            detail="Receipt ID already exists with different content",
        )

    existing_run = load_run(run_id)
    if not existing_run:
        raise HTTPException(status_code=404, detail="Run not found")

    state = existing_run["state"]

    for outcome in body.outcomes:
        if state.get("stage") in ("completed", "failed"):
            raise HTTPException(
                status_code=422,
                detail="Run already in terminal state",
            )
        try:
            pending = state.get("pendingActions", {})
            if outcome.actionId in pending:
                info = pending[outcome.actionId]
                if info["phase"] == "effect":
                    state = process_effect_outcome(state, outcome, body.receiptId)
                else:
                    state = process_outcome(state, outcome, body.receiptId, None)
            else:
                raise ValueError(f"Unknown action: {outcome.actionId}")
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    for approval in body.approvals:
        if state.get("stage") in ("completed", "failed"):
            raise HTTPException(
                status_code=422,
                detail="Run already in terminal state",
            )
        try:
            state = process_approval(state, approval, body.receiptId)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e))

    state["otlp"] = rebuild_otlp(state)
    if isinstance(state.get("currentResponse"), dict):
        state["currentResponse"]["otlp"] = state["otlp"]
        state["currentResponse"]["actionLog"] = state.get("actionLog", [])
        state["currentResponse"]["receiptLog"] = state.get("receiptLog", [])

    save_run(run_id, existing_run["request_hash"], state)
    save_receipt(run_id, body.receiptId, receipt_hash, state.get("currentResponse", {}))

    return state.get("currentResponse", {})


@router.get("/v2/incidents/{run_id}")
def get_incident(run_id: str):
    existing = load_run(run_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Run not found")
    return existing["state"]["currentResponse"]


FORBIDDEN_TELEMETRY_KEYS = {
    "transcript",
    "prompt",
    "sensitive",
    "accessToken",
    "privateNote",
    "authorization",
    "arguments",
    "result",
    "observation",
    "body",
    "gen_ai.tool.call.arguments",
    "gen_ai.tool.call.result",
}


def _redact_check(response: dict, request: CreateIncidentRequest) -> None:
    from q11_utils import canonical_json

    serialised = canonical_json(response)

    secret_values = []
    if request.sensitive.accessToken:
        secret_values.append(request.sensitive.accessToken)
    if request.sensitive.privateNote:
        secret_values.append(request.sensitive.privateNote)

    for secret in secret_values:
        if secret and secret in serialised:
            raise RuntimeError("Sensitive value found in exported state")
