from __future__ import annotations

import time
from typing import Any

from q11_models import SPAN_KIND_INTERNAL, SPAN_KIND_SERVER, SPAN_KIND_CLIENT
from q11_utils import new_span_id


def otlp_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def otlp_attributes(values: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"key": key, "value": otlp_value(value)}
        for key, value in values.items()
    ]


def now_unix_nano() -> str:
    return str(time.time_ns())


def make_span(
    *,
    trace_id: str,
    span_id: str,
    parent_span_id: str | None,
    name: str,
    kind: int,
    attributes: dict[str, Any],
    links: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    start = time.time_ns()
    span: dict[str, Any] = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name,
        "kind": kind,
        "startTimeUnixNano": str(start),
        "endTimeUnixNano": str(start + 1_000_000),
        "attributes": otlp_attributes(attributes),
        "status": {"code": 0},
    }
    if parent_span_id:
        span["parentSpanId"] = parent_span_id
    if links:
        span["links"] = links
    return span


def update_span_end_time(span: dict[str, Any]) -> None:
    span["endTimeUnixNano"] = str(time.time_ns())


def make_resource_spans(
    trace_id: str,
    spans: list[dict[str, Any]],
    run_id: str,
    public_marker: str,
) -> dict[str, Any]:
    for span in spans:
        has_run_id = any(
            attr.get("key") == "ga5.run.id" for attr in span.get("attributes", [])
        )
        if not has_run_id:
            span["attributes"].extend(
                otlp_attributes({
                    "ga5.run.id": run_id,
                    "ga5.public.marker": public_marker,
                })
            )
    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {
                            "key": "service.name",
                            "value": {"stringValue": "incident-response"},
                        }
                    ]
                },
                "scopeSpans": [
                    {
                        "scope": {
                            "name": "ga5.incident-agent",
                            "version": "2.0.0",
                        },
                        "spans": spans,
                    }
                ],
            }
        ]
    }
