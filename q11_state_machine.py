from __future__ import annotations

from typing import Any

from q11_models import (
    ApprovalDecision,
    CreateIncidentRequest,
    Outcome,
    PlannerOutput,
    ReceiptRequest,
)
from q11_utils import (
    canonical_json,
    new_span_id,
    opaque_id,
    parse_traceparent,
    sha256_hex,
)
from q11_telemetry import (
    make_span,
    make_resource_spans,
    otlp_attributes,
    update_span_end_time,
)



def build_initial_state(
    request: CreateIncidentRequest,
    request_hash: str,
    plan: PlannerOutput,
    incoming_trace_id: str,
    incoming_span_id: str | None,
    incoming_tracestate: str | None,
    model_name: str = "gpt-4",
) -> dict[str, Any]:
    run_id = request.runId
    public_marker = request.publicMarker
    trace_id = incoming_trace_id

    server_span_id = new_span_id()
    agent_span_id = new_span_id()
    model_span_id = new_span_id()

    server_span = make_span(
        trace_id=trace_id,
        span_id=server_span_id,
        parent_span_id=incoming_span_id,
        name="POST /v2/incidents",
        kind=2,
        attributes={
            "ga5.run.id": run_id,
            "ga5.public.marker": public_marker,
        },
    )

    agent_span = make_span(
        trace_id=trace_id,
        span_id=agent_span_id,
        parent_span_id=server_span_id,
        name="invoke_agent incident-response",
        kind=1,
        attributes={
            "ga5.run.id": run_id,
            "ga5.public.marker": public_marker,
        },
    )

    model_span = make_span(
        trace_id=trace_id,
        span_id=model_span_id,
        parent_span_id=agent_span_id,
        name="chat incident-plan",
        kind=3,
        attributes={
            "ga5.run.id": run_id,
            "ga5.public.marker": public_marker,
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": model_name,
        },
    )

    all_spans = [server_span, agent_span, model_span]
    action_log: list[dict[str, Any]] = []
    dispatch_list: list[dict[str, Any]] = []
    execute_span_ids: list[str] = []

    exec_span_map: dict[str, str] = {}

    for i, diag in enumerate(plan.diagnostics):
        action_id = opaque_id("act")
        call_id = opaque_id("call")
        client_span_id = new_span_id()

        execute_span_id = new_span_id()
        execute_span_ids.append(execute_span_id)
        exec_span_map[action_id] = execute_span_id

        tool_name = diag["toolName"]
        execute_span = make_span(
            trace_id=trace_id,
            span_id=execute_span_id,
            parent_span_id=agent_span_id,
            name=f"execute_tool {tool_name}",
            kind=1,
            attributes={
                "ga5.action.id": action_id,
                "gen_ai.tool.name": tool_name,
                "gen_ai.tool.call.id": call_id,
                "gen_ai.operation.name": "execute_tool",
            },
        )
        all_spans.append(execute_span)

        client_span = make_span(
            trace_id=trace_id,
            span_id=client_span_id,
            parent_span_id=execute_span_id,
            name=f"POST tool/{tool_name}",
            kind=3,
            attributes={
                "ga5.run.id": run_id,
                "ga5.public.marker": public_marker,
                "ga5.action.id": action_id,
                "ga5.attempt": 1,
            },
        )
        all_spans.append(client_span)

        traceparent = f"00-{trace_id}-{client_span_id}-01"

        dispatch = {
            "actionId": action_id,
            "callId": call_id,
            "phase": "diagnostic",
            "toolName": tool_name,
            "arguments": diag["arguments"],
            "evidence": diag["evidence"],
            "attempt": 1,
            "traceparent": traceparent,
        }
        if incoming_tracestate:
            dispatch["tracestate"] = incoming_tracestate
        action_log.append(dispatch)
        dispatch_list.append(dispatch)

    join_span_id = new_span_id()
    join_span = make_span(
        trace_id=trace_id,
        span_id=join_span_id,
        parent_span_id=agent_span_id,
        name="incident.join",
        kind=1,
        attributes={
            "ga5.run.id": run_id,
            "ga5.public.marker": public_marker,
        },
        links=[
            {
                "traceId": trace_id,
                "spanId": eid,
                "attributes": [],
            }
            for eid in execute_span_ids
        ],
    )

    effect_tool = plan.effectPlan["toolName"]
    needs_approval = effect_tool in request.policy.approvalRequiredFor

    state = {
        "runId": run_id,
        "agentName": request.agentName,
        "publicMarker": public_marker,
        "requestHash": request_hash,
        "incomingTraceparent": incoming_trace_id,
        "incomingTracestate": incoming_tracestate,
        "incident": request.incident.model_dump(),
        "policy": request.policy.model_dump(),
        "plan": plan.model_dump(),
        "stage": "diagnostics_pending",
        "diagnosis": {
            "rootCause": plan.rootCause,
            "evidence": plan.evidence,
        },
        "chosenEffect": None,
        "suppressed": [],
        "actionLog": list(action_log),
        "receiptLog": [],
        "pendingActions": {
            d["actionId"]: {
                "callId": d["callId"],
                "attempt": d["attempt"],
                "phase": d["phase"],
                "toolName": d["toolName"],
            }
            for d in action_log
        },
        "approvalPending": None,
        "effectDispatched": False,
        "reservedEffectActionId": None,
        "reservedEffectCallId": None,
        "reservedApprovalId": None,
        "execSpanMap": exec_span_map,
        "spans": [],
        "currentResponse": {},
    }

    if len(execute_span_ids) > 1:
        all_spans.append(join_span)
    state["spans"] = all_spans

    state["currentResponse"] = {
        "runId": run_id,
        "status": "waiting",
        "diagnosis": {
            "rootCause": plan.rootCause,
            "evidence": plan.evidence,
        },
        "dispatches": dispatch_list,
        "approvals": [],
        "actionLog": list(action_log),
        "receiptLog": [],
        "otlp": make_resource_spans(trace_id, all_spans, run_id, public_marker),
        "chosenEffect": None,
        "suppressed": [],
    }

    return state


def _make_execute_span_from_dispatch(
    dispatch: dict, trace_id: str, parent_id: str, run_id: str, marker: str,
) -> dict:
    return make_span(
        trace_id=trace_id,
        span_id=new_span_id(),
        parent_span_id=parent_id,
        name=f"execute_tool {dispatch['toolName']}",
        kind=1,
        attributes={
            "ga5.action.id": dispatch["actionId"],
            "gen_ai.tool.name": dispatch["toolName"],
            "gen_ai.tool.call.id": dispatch["callId"],
            "gen_ai.operation.name": "execute_tool",
        },
    )


def _make_client_spans_from_dispatch(
    dispatch: dict, trace_id: str, parent_id: str, run_id: str, marker: str,
) -> list[dict]:
    client_span_id = new_span_id()
    client_span = make_span(
        trace_id=trace_id,
        span_id=client_span_id,
        parent_span_id=parent_id,
        name=f"POST tool/{dispatch['toolName']}",
        kind=3,
        attributes={
            "ga5.run.id": run_id,
            "ga5.public.marker": marker,
            "ga5.action.id": dispatch["actionId"],
            "ga5.attempt": dispatch["attempt"],
        },
    )
    return [client_span]


def _build_waiting_response(state: dict) -> dict:
    cr = state.get("currentResponse", {})
    return {
        "runId": state["runId"],
        "status": "waiting",
        "diagnosis": state["diagnosis"],
        "dispatches": cr.get("dispatches", state.get("dispatches", [])),
        "approvals": cr.get("approvals", state.get("approvals", [])),
        "actionLog": cr.get("actionLog", state.get("actionLog", [])),
        "receiptLog": cr.get("receiptLog", state.get("receiptLog", [])),
        "otlp": cr.get("otlp", state.get("otlp", {})),
        "chosenEffect": cr.get("chosenEffect", state.get("chosenEffect")),
        "suppressed": cr.get("suppressed", state.get("suppressed", [])),
    }


def _build_failed_response(state: dict) -> dict:
    return {
        "runId": state["runId"],
        "status": "failed",
        "diagnosis": state["diagnosis"],
        "dispatches": [],
        "approvals": [],
        "actionLog": state.get("actionLog", []),
        "receiptLog": state.get("receiptLog", []),
        "otlp": state.get("otlp", {}),
        "chosenEffect": state.get("chosenEffect"),
        "suppressed": state.get("suppressed", []),
    }


def _build_completed_response(state: dict) -> dict:
    return {
        "runId": state["runId"],
        "status": "completed",
        "diagnosis": state["diagnosis"],
        "dispatches": [],
        "approvals": [],
        "actionLog": state.get("actionLog", []),
        "receiptLog": state.get("receiptLog", []),
        "otlp": state.get("otlp", {}),
        "chosenEffect": state.get("chosenEffect"),
        "suppressed": state.get("suppressed", []),
    }


def process_outcome(
    state: dict,
    outcome: Outcome,
    receipt_id: str,
    request: CreateIncidentRequest | None,
) -> dict:
    action_id = outcome.actionId
    call_id = outcome.callId
    attempt = outcome.attempt
    status = outcome.status

    pending = state.get("pendingActions", {})
    if action_id not in pending:
        raise ValueError(f"Unknown action: {action_id}")

    pending_info = pending[action_id]
    if pending_info["callId"] != call_id:
        raise ValueError(f"callId mismatch for action {action_id}")
    if pending_info["attempt"] != attempt:
        raise ValueError(f"attempt mismatch for action {action_id}: expected {pending_info['attempt']}, got {attempt}")

    flat_receipt = {
        "receiptId": receipt_id,
        "actionId": action_id,
        "callId": call_id,
        "attempt": attempt,
        "status": status,
        "resultClass": outcome.resultClass,
        "nonce": outcome.nonce,
    }
    # A zero HTTP status is ambiguous without the authoritative error class.
    # Persist it so the receipt log and CLIENT span describe the same timeout.
    if outcome.errorType:
        flat_receipt["errorType"] = outcome.errorType
    if "receiptLog" not in state:
        state["receiptLog"] = []
    state["receiptLog"].append(flat_receipt)

    _update_client_span(state, action_id, attempt, receipt_id, outcome)

    if status == 503:
        if attempt == 1:
            return _handle_503_retry(state, action_id, call_id, receipt_id, outcome)
        del pending[action_id]
        return _handle_diagnostic_failure(state, action_id, "diagnostic_failed")
    elif status == 0 or outcome.errorType == "timeout":
        del pending[action_id]
        return _handle_diagnostic_failure(state, action_id, "diagnostic_timed_out")
    elif status == 200 or status == 201:
        del pending[action_id]
        return _check_all_diagnostics_done(state)
    else:
        del pending[action_id]
        return _handle_diagnostic_failure(state, action_id, "diagnostic_failed")


def _update_client_span(
    state: dict, action_id: str, attempt: int, receipt_id: str, outcome: Outcome,
) -> None:
    for span in state.get("spans", []):
        attrs = {a["key"]: _otlp_attr_value(a.get("value", {})) for a in span.get("attributes", [])}
        if (
            attrs.get("ga5.action.id") == action_id
            and attrs.get("ga5.attempt") == attempt
            and span.get("kind") == 3
        ):
            span["attributes"].extend(
                otlp_attributes({
                    "ga5.receipt.id": receipt_id,
                    "ga5.receipt.nonce": outcome.nonce or "",
                    "http.request.method": "POST",
                    "http.request.resend_count": attempt - 1,
                    "http.response.status_code": outcome.status,
                })
            )
            if outcome.status == 503 or (outcome.status == 0 and outcome.errorType == "timeout"):
                span["status"] = {"code": 2}
                if outcome.status == 503:
                    span["attributes"].append(
                        {"key": "error.type", "value": {"stringValue": "503"}}
                    )
                elif outcome.errorType == "timeout":
                    span["attributes"].append(
                        {"key": "error.type", "value": {"stringValue": "timeout"}}
                    )
            else:
                span["status"] = {"code": 0}
            update_span_end_time(span)
            break


def _handle_503_retry(
    state: dict, action_id: str, call_id: str, receipt_id: str, outcome: Outcome,
) -> dict:
    pending = state["pendingActions"]
    info = pending[action_id]
    info["attempt"] = 2

    tool_name = info["toolName"]

    new_client_span_id = new_span_id()
    new_client_span = make_span(
        trace_id=state["incomingTraceparent"],
        span_id=new_client_span_id,
        parent_span_id=_get_execute_span_id(state, action_id),
        name=f"POST tool/{tool_name}",
        kind=3,
        attributes={
            "ga5.run.id": state["runId"],
            "ga5.public.marker": state["publicMarker"],
            "ga5.action.id": action_id,
            "ga5.attempt": 2,
            "http.request.resend_count": 1,
        },
    )
    state["spans"].append(new_client_span)

    traceparent = f"00-{state['incomingTraceparent']}-{new_client_span_id}-01"

    retry_dispatch = {
        "actionId": action_id,
        "callId": call_id,
        "phase": info["phase"],
        "toolName": tool_name,
        "arguments": _find_dispatch_args(state, action_id),
        "evidence": _find_dispatch_evidence(state, action_id),
        "attempt": 2,
        "traceparent": traceparent,
    }
    if state.get("incomingTracestate"):
        retry_dispatch["tracestate"] = state["incomingTracestate"]
    state["actionLog"].append(retry_dispatch)

    state["currentResponse"] = {
        "runId": state["runId"],
        "status": "waiting",
        "diagnosis": state["diagnosis"],
        "dispatches": [retry_dispatch],
        "approvals": [],
        "actionLog": state["actionLog"],
        "receiptLog": state["receiptLog"],
        "otlp": state.get("otlp", {}),
        "chosenEffect": state.get("chosenEffect"),
        "suppressed": state.get("suppressed", []),
    }
    return state


def _get_execute_span_id(state: dict, action_id: str) -> str | None:
    for span in state.get("spans", []):
        attrs = {a["key"]: _otlp_attr_value(a.get("value", {})) for a in span.get("attributes", [])}
        if attrs.get("ga5.action.id") == action_id and span.get("kind") == 1:
            return span["spanId"]
    return None


def _otlp_attr_value(value: dict[str, Any]) -> Any:
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    return None


def _find_dispatch_args(state: dict, action_id: str) -> dict:
    for d in state.get("actionLog", []):
        if d["actionId"] == action_id:
            return d.get("arguments", {})
    return {}


def _find_dispatch_evidence(state: dict, action_id: str) -> list:
    for d in state.get("actionLog", []):
        if d["actionId"] == action_id:
            return d.get("evidence", [])
    return []


def _handle_diagnostic_failure(state: dict, action_id: str, reason: str) -> dict:
    effect_tool = state["plan"]["effectPlan"]["toolName"]
    # The public result records which dependent effect was suppressed. The
    # failure reason is already represented by the authoritative receipt and
    # the ERROR client span.
    state["suppressed"] = [effect_tool]
    state["stage"] = "failed"
    state["currentResponse"] = _build_failed_response(state)
    return state


def _check_all_diagnostics_done(state: dict) -> dict:
    pending = state.get("pendingActions", {})
    active_diags = {
        k: v for k, v in pending.items() if v["phase"] == "diagnostic"
    }
    if active_diags:
        state["currentResponse"] = _build_waiting_response(state)
        return state

    effect_tool = state["plan"]["effectPlan"]["toolName"]
    effect_args = state["plan"]["effectPlan"]["arguments"]
    from q11_utils import opaque_id as _opaque

    policy = state.get("policy", {})
    needs_approval = effect_tool in policy.get("approvalRequiredFor", [])

    if needs_approval:
        effect_action_id = _opaque("act")
        effect_call_id = _opaque("call")
        approval_id = _opaque("approval")
        digest = sha256_hex(effect_args)

        state["reservedEffectActionId"] = effect_action_id
        state["reservedEffectCallId"] = effect_call_id
        state["reservedApprovalId"] = approval_id
        state["stage"] = "approval_pending"
        state["approvalPending"] = {
            "approvalId": approval_id,
            "actionId": effect_action_id,
            "toolName": effect_tool,
            "argumentsDigest": digest,
        }

        state["currentResponse"] = {
            "runId": state["runId"],
            "status": "waiting",
            "diagnosis": state["diagnosis"],
            "dispatches": [],
            "approvals": [
                {
                    "approvalId": approval_id,
                    "actionId": effect_action_id,
                    "toolName": effect_tool,
                    "argumentsDigest": digest,
                }
            ],
            "actionLog": state.get("actionLog", []),
            "receiptLog": state.get("receiptLog", []),
            "otlp": state.get("otlp", {}),
            "chosenEffect": state.get("chosenEffect"),
            "suppressed": state.get("suppressed", []),
        }
        return state

    return _dispatch_effect(state)


def _dispatch_effect(state: dict) -> dict:
    effect_tool = state["plan"]["effectPlan"]["toolName"]
    effect_args = state["plan"]["effectPlan"]["arguments"]
    from q11_utils import opaque_id as _opaque

    action_id = state.get("reservedEffectActionId") or _opaque("act")
    call_id = state.get("reservedEffectCallId") or _opaque("call")
    approval_id = state.get("reservedApprovalId")
    approval_obj = state.get("approvalPending")

    evidence = state["diagnosis"]["evidence"]

    new_client_span_id = new_span_id()
    execute_span_id = new_span_id()

    trace_id = state["incomingTraceparent"]
    agent_span_id = _get_agent_span_id(state)
    run_id = state["runId"]
    marker = state["publicMarker"]

    execute_span = make_span(
        trace_id=trace_id,
        span_id=execute_span_id,
        parent_span_id=agent_span_id,
        name=f"execute_tool {effect_tool}",
        kind=1,
        attributes={
            "ga5.action.id": action_id,
            "gen_ai.tool.name": effect_tool,
            "gen_ai.tool.call.id": call_id,
            "gen_ai.operation.name": "execute_tool",
        },
    )
    state["spans"].append(execute_span)

    client_span = make_span(
        trace_id=trace_id,
        span_id=new_client_span_id,
        parent_span_id=execute_span_id,
        name=f"POST tool/{effect_tool}",
        kind=3,
        attributes={
            "ga5.run.id": run_id,
            "ga5.public.marker": marker,
            "ga5.action.id": action_id,
            "ga5.attempt": 1,
        },
    )
    state["spans"].append(client_span)

    traceparent = f"00-{trace_id}-{new_client_span_id}-01"

    dispatch = {
        "actionId": action_id,
        "callId": call_id,
        "phase": "effect",
        "toolName": effect_tool,
        "arguments": effect_args,
        "evidence": evidence,
        "attempt": 1,
        "traceparent": traceparent,
    }
    if state.get("incomingTracestate"):
        dispatch["tracestate"] = state["incomingTracestate"]

    if approval_id:
        dispatch["approvalId"] = approval_id
        approval_obj = state.get("approvalPending", {})
        dispatch["approvalNonce"] = approval_obj.get("nonce", "")

    state["actionLog"].append(dispatch)
    state["pendingActions"][action_id] = {
        "callId": call_id,
        "attempt": 1,
        "phase": "effect",
        "toolName": effect_tool,
    }
    state["stage"] = "effect_pending"
    state["effectDispatched"] = True

    state["currentResponse"] = {
        "runId": run_id,
        "status": "waiting",
        "diagnosis": state["diagnosis"],
        "dispatches": [dispatch],
        "approvals": [],
        "actionLog": state["actionLog"],
        "receiptLog": state["receiptLog"],
        "otlp": state.get("otlp", {}),
        "chosenEffect": state.get("chosenEffect"),
        "suppressed": state.get("suppressed", []),
    }
    return state


def _get_agent_span_id(state: dict) -> str | None:
    for span in state.get("spans", []):
        if span.get("name") == "invoke_agent incident-response":
            return span["spanId"]
    return None


def process_approval(
    state: dict,
    approval: ApprovalDecision,
    receipt_id: str,
) -> dict:
    pending_approval = state.get("approvalPending")
    if not pending_approval:
        raise ValueError("No pending approval")

    if pending_approval["approvalId"] != approval.approvalId:
        raise ValueError("approvalId mismatch")

    flat = {
        "receiptId": receipt_id,
        "approvalId": approval.approvalId,
        "decision": approval.decision,
        "nonce": approval.nonce,
    }
    state.setdefault("receiptLog", []).append(flat)

    approval_span = make_span(
        trace_id=state["incomingTraceparent"],
        span_id=new_span_id(),
        parent_span_id=_get_agent_span_id(state),
        name="approval_gate",
        kind=1,
        attributes={
            "ga5.run.id": state["runId"],
            "ga5.public.marker": state["publicMarker"],
            "ga5.approval.id": approval.approvalId,
            "ga5.receipt.id": receipt_id,
            "ga5.receipt.nonce": approval.nonce,
        },
    )
    state["spans"].append(approval_span)

    if approval.decision == "approved":
        state["stage"] = "effect_pending"
        pending_approval["nonce"] = approval.nonce
        return _dispatch_effect(state)
    else:
        effect_tool = state["plan"]["effectPlan"]["toolName"]
        state["suppressed"] = [effect_tool]
        state["stage"] = "failed"
        state["currentResponse"] = _build_failed_response(state)
        return state


def process_effect_outcome(
    state: dict,
    outcome: Outcome,
    receipt_id: str,
) -> dict:
    action_id = outcome.actionId
    pending = state.get("pendingActions", {})
    if action_id not in pending:
        raise ValueError(f"Unknown effect action: {action_id}")

    info = pending[action_id]
    if info["callId"] != outcome.callId:
        raise ValueError("callId mismatch for effect")
    if info["attempt"] != outcome.attempt:
        raise ValueError("attempt mismatch for effect")

    flat = {
        "receiptId": receipt_id,
        "actionId": action_id,
        "callId": outcome.callId,
        "attempt": outcome.attempt,
        "status": outcome.status,
        "resultClass": outcome.resultClass,
        "nonce": outcome.nonce,
    }
    if outcome.errorType:
        flat["errorType"] = outcome.errorType
    state.setdefault("receiptLog", []).append(flat)

    _update_client_span(state, action_id, outcome.attempt, receipt_id, outcome)

    if outcome.status == 503 and outcome.attempt == 1:
        return _handle_503_retry(state, action_id, outcome.callId, receipt_id, outcome)

    del pending[action_id]

    if outcome.status in (200, 201):
        state["stage"] = "completed"
        state["chosenEffect"] = state["plan"]["effectPlan"]["toolName"]
        state["currentResponse"] = _build_completed_response(state)
        return state

    state["stage"] = "failed"
    state["suppressed"] = [state["plan"]["effectPlan"]["toolName"]]
    state["currentResponse"] = _build_failed_response(state)
    return state


def rebuild_otlp(state: dict) -> dict:
    trace_id = state.get("incomingTraceparent", new_span_id())
    run_id = state["runId"]
    marker = state["publicMarker"]
    return make_resource_spans(trace_id, state.get("spans", []), run_id, marker)
