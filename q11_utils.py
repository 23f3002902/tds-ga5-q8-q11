from __future__ import annotations

import hashlib
import json
import re
import secrets
from typing import Any

from q11_models import TRACEPARENT_RE


EVIDENCE_PATTERN = re.compile(
    r"^\[([A-Za-z0-9_.:-]+)\]\s*(.*)$",
    re.MULTILINE,
)


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_hex(value: Any) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def opaque_id(prefix: str) -> str:
    return f"{prefix}_{secrets.token_hex(12)}"


def parse_traceparent(value: str) -> tuple[str, str, str]:
    match = TRACEPARENT_RE.fullmatch(value)
    if not match:
        raise ValueError("Invalid traceparent")
    trace_id, parent_span_id, flags = match.groups()
    if trace_id == "0" * 32:
        raise ValueError("Zero trace ID")
    if parent_span_id == "0" * 16:
        raise ValueError("Zero parent span ID")
    return trace_id, parent_span_id, flags


def new_trace_id() -> str:
    while True:
        value = secrets.token_hex(16)
        if value != "0" * 32:
            return value


def new_span_id() -> str:
    while True:
        value = secrets.token_hex(8)
        if value != "0" * 16:
            return value


def extract_evidence_lines(transcript: str) -> dict[str, str]:
    evidence: dict[str, str] = {}
    for match in EVIDENCE_PATTERN.finditer(transcript):
        evidence_id = match.group(1)
        text = match.group(2).strip()
        evidence[evidence_id] = text
    return evidence
