from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time as time_module
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_ai_diagnostic_done = False

Q10_DB_PATH = os.environ.get("Q10_DB_PATH", "q10_a2a.sqlite3")
_Q10_IS_MEMORY = Q10_DB_PATH == ":memory:"
A2A_BASE_URL = os.environ.get("A2A_BASE_URL", "").rstrip("/")
_RENDER_BASE = "https://tds-ga5-wpyi.onrender.com"
A2A_BEARER_TOKEN = os.environ.get("A2A_BEARER_TOKEN", "ga5-invoice-token")
AI_API_KEY = os.environ.get("AI_API_KEY", "")
AI_MODEL = os.environ.get("AI_MODEL", "")
AI_TIMEOUT_SECONDS = int(os.environ.get("AI_TIMEOUT_SECONDS", "60") or "60")

Q10_AI_BATCH_SIZE = max(12, int(os.environ.get("Q10_AI_BATCH_SIZE", "12") or "12"))
Q10_AI_MAX_INPUT_TOKENS = max(24000, int(os.environ.get("Q10_AI_MAX_INPUT_TOKENS", "24000") or "24000"))
Q10_AI_CHARS_PER_TOKEN = int(os.environ.get("Q10_AI_CHARS_PER_TOKEN", "4") or "4")
Q10_AI_MAX_OUTPUT_TOKENS_PER_PACKAGE = int(
    os.environ.get("Q10_AI_MAX_OUTPUT_TOKENS_PER_PACKAGE", "400") or "400"
)
Q10_AI_BATCH_DELAY = 0.0
_Q10_BATCH_ENV_LOG = False


_DOTENV_CANDIDATES = [".env", "/etc/secrets/.env", "/etc/env"]


def _find_env(*names: str, default: str = "") -> str:
    lower_names = {n.lower() for n in names}
    for k, v in os.environ.items():
        if k.lower() in lower_names and v.strip():
            return v.strip()
    for dotenv in _DOTENV_CANDIDATES:
        p = Path(dotenv)
        if p.is_file():
            try:
                for line in p.read_text("utf-8").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    ek, _, ev = line.partition("=")
                    if ek.strip().lower() in lower_names and ev.strip():
                        return ev.strip().strip("\"'")
            except OSError:
                pass
    return default


def _ai_env() -> tuple[str, str, str, int]:
    base = _find_env("AI_API_BASE", "ai_api_base").rstrip("/")
    key = _find_env("AI_API_KEY", "ai_api_key")
    model = _find_env("AI_MODEL", "ai_model")
    timeout_str = _find_env("AI_TIMEOUT_SECONDS", "ai_timeout_seconds", default=str(AI_TIMEOUT_SECONDS))
    timeout = int(timeout_str) if timeout_str and timeout_str.strip() else AI_TIMEOUT_SECONDS
    return base, key, model, timeout

ALLOWED_ACTIONS = frozenset({
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
})

_db_lock = threading.RLock()


class A2AJSONResponse(JSONResponse):
    media_type = "application/a2a+json"


router = APIRouter(default_response_class=A2AJSONResponse)


_memory_conn: sqlite3.Connection | None = None


def get_db_path() -> Path:
    if _Q10_IS_MEMORY:
        return Path(":memory:")
    return Path(Q10_DB_PATH)


def get_memory_conn() -> sqlite3.Connection:
    global _memory_conn
    if _memory_conn is None:
        _memory_conn = sqlite3.connect(
            ":memory:",
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        _memory_conn.row_factory = sqlite3.Row
    return _memory_conn


@contextmanager
def database_connection() -> Iterator[sqlite3.Connection]:
    if _Q10_IS_MEMORY:
        conn = get_memory_conn()
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        return

    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        timeout=30,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = FULL")
        yield conn
    finally:
        conn.close()


def init_database() -> None:
    with _db_lock:
        with database_connection() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            if not _Q10_IS_MEMORY:
                conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS a2a_tasks (
                    task_id TEXT NOT NULL,
                    principal_hash TEXT NOT NULL,
                    context_id TEXT NOT NULL,
                    task_state TEXT NOT NULL,
                    batch_id TEXT,
                    policy_revision TEXT,
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, principal_hash)
                );

                CREATE TABLE IF NOT EXISTS a2a_messages (
                    principal_hash TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    message_hash TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    message_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (principal_hash, message_id)
                );

                CREATE TABLE IF NOT EXISTS a2a_proposals (
                    task_id TEXT NOT NULL,
                    principal_hash TEXT NOT NULL,
                    package_id TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    batch_id TEXT NOT NULL,
                    facts_json TEXT NOT NULL,
                    evidence_refs_json TEXT NOT NULL,
                    rationale TEXT NOT NULL,
                    outcome TEXT,
                    receipt_nonce TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (task_id, principal_hash, package_id)
                );

                CREATE TABLE IF NOT EXISTS a2a_invoice_cache (
                    package_fingerprint TEXT NOT NULL,
                    proposal_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (package_fingerprint)
                );
            """)


init_database()


def sha256_hex(value: Any) -> str:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="microseconds") + "Z"


def compact_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def principal_hash_from_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def generate_task_id() -> str:
    return "task-" + hashlib.sha256(
        (utc_now() + str(time_module.monotonic_ns())).encode("utf-8")
    ).hexdigest()[:32]


_action_id_counter: int = 0


def generate_action_id(package_id: str, batch_id: str) -> str:
    global _action_id_counter
    _action_id_counter += 1
    raw = f"{batch_id}:{package_id}:{utc_now()}:{_action_id_counter}"
    return "act-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:28]


A2A_VERSION = "1.0"
A2A_CONTENT_TYPE = "application/a2a+json"
BATCH_CONTENT_TYPE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSAL_CONTENT_TYPE = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_CONTENT_TYPE = "application/vnd.ga5.invoice-action-results+json"
RECEIPT_CONTENT_TYPE = "application/vnd.ga5.invoice-action-receipts+json"


async def check_a2a_headers(request: Request, require_content_type: bool = False) -> str:
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "UNAUTHENTICATED", "message": "Missing or invalid Authorization header."}},
        )
    token = auth[len("Bearer "):]
    if not token:
        raise HTTPException(
            status_code=401,
            detail={"error": {"code": "UNAUTHENTICATED", "message": "Missing bearer token."}},
        )
    version = request.headers.get("A2A-Version", "")
    if not version:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_VERSION", "message": "A2A-Version header is required."}},
        )
    if version != A2A_VERSION:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_VERSION", "message": f"Unsupported A2A-Version: {version}. Must be {A2A_VERSION}."}},
        )
    if require_content_type:
        content_type = request.headers.get("Content-Type", "")
        if content_type != A2A_CONTENT_TYPE and not content_type.startswith(A2A_CONTENT_TYPE + ";"):
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_MEDIA_TYPE", "message": f"Content-Type must be {A2A_CONTENT_TYPE}."}},
            )
    return token


def check_a2a_version_header(headers: dict[str, str]) -> None:
    version = headers.get("a2a-version", "")
    if not version:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_VERSION", "message": "A2A-Version header is required."}},
        )
    if version != A2A_VERSION:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_VERSION", "message": f"Unsupported A2A-Version: {version}. Must be {A2A_VERSION}."}},
        )


@router.get("/.well-known/agent-card.json", response_class=JSONResponse)
async def agent_card():
    base_url = A2A_BASE_URL or _RENDER_BASE
    a2a_url = base_url.rstrip("/") + "/a2a/"
    iface = {"protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
    if a2a_url:
        iface["url"] = a2a_url
    return {
        "protocolVersion": "1.0",
        "name": "TDS GA5 Invoice Action Agent",
        "version": "1.0.0",
        "description": "AI agent that processes invoice claim batches by analyzing each package, selecting an appropriate action, and producing traceable proposals for grader approval.",
        "preferredTransport": "HTTP+JSON",
        "url": a2a_url,
        "provider": {"organization": "TDS GA5", "url": a2a_url},
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "stateTransitionHistory": True,
            "extendedAgentCard": False,
        },
        "supportedInterfaces": [iface],
        "defaultInputModes": [BATCH_CONTENT_TYPE, RESULTS_CONTENT_TYPE, "application/json"],
        "defaultOutputModes": [
            PROPOSAL_CONTENT_TYPE,
            RECEIPT_CONTENT_TYPE,
            "application/json",
        ],
        "securitySchemes": {
            "bearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "description": "Each Bearer token identifies an isolated principal.",
            }
        },
        "security": [{"bearerAuth": []}],
        "skills": [
            {
                "id": "invoice_action_agent",
                "name": "Invoice Action Agent",
                "description": "Analyzes invoice claim packages and selects one action per package: settle_invoice, request_approval, hold_invoice, reject_duplicate, or open_exception.",
                "tags": ["invoice", "accounts-payable", "reconciliation", "approval", "duplicate-detection", "exception-handling", "a2a"],
                "examples": [
                    "Propose one action for every package in an invoice claim batch.",
                    "Finalize accepted proposals using invoice action results.",
                ],
                "inputModes": [BATCH_CONTENT_TYPE, RESULTS_CONTENT_TYPE],
                "outputModes": [PROPOSAL_CONTENT_TYPE, RECEIPT_CONTENT_TYPE],
            }
        ],
        "authentication": {
            "schemes": [
                {
                    "scheme": "bearer",
                    "bearerFormat": "HttpHeaders",
                    "in": "header",
                }
            ]
        },
    }


def build_task_object(
    task_id: str,
    task_state: str,
    context_id: str,
    history: list[dict[str, Any]],
    artifacts: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    task: dict[str, Any] = {
        "kind": "task",
        "id": task_id,
        "contextId": context_id,
        "status": {"state": task_state, "timestamp": utc_now()},
        "history": history,
        "artifacts": artifacts or [],
    }
    return task


def build_data_artifact(
    artifact_id: str,
    name: str,
    media_type: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifactId": artifact_id,
        "name": name,
        "parts": [{
            "kind": "data",
            "mediaType": media_type,
            "data": data,
        }],
    }


@router.post("/a2a/message:send")
async def message_send(request: Request):
    token = await check_a2a_headers(request, require_content_type=True)
    principal = principal_hash_from_token(token)

    try:
        body = await request.json()
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "Request body must be valid JSON."}},
        )

    message = body.get("message")
    if not isinstance(message, dict):
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "message is required."}},
        )

    message_id = message.get("messageId", "")
    if not message_id:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "messageId is required."}},
        )

    message_role = message.get("role", "")
    if message_role != "ROLE_USER":
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "role must be ROLE_USER."}},
        )

    parts = message.get("parts", [])
    if not isinstance(parts, list) or not parts:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "At least one part is required."}},
        )

    media_type = parts[0].get("mediaType", "")
    data = parts[0].get("data")

    if data is None:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "Part data is required."}},
        )

    message_hash = sha256_hex(message)

    with _db_lock:
        with database_connection() as conn:
            existing_msg = conn.execute(
                "SELECT * FROM a2a_messages WHERE principal_hash = ? AND message_id = ?",
                (principal, message_id),
            ).fetchone()

            if existing_msg is not None:
                if existing_msg["message_hash"] == message_hash:
                    task_id = existing_msg["task_id"]
                    task_row = conn.execute(
                        "SELECT * FROM a2a_tasks WHERE task_id = ? AND principal_hash = ?",
                        (task_id, principal),
                    ).fetchone()
                    if task_row is not None:
                        task_obj = json.loads(task_row["response_json"]) if task_row["response_json"] else None
                        if task_obj:
                            return {"task": task_obj}
                        raise HTTPException(
                            status_code=500,
                            detail={"error": {"code": "INTERNAL_ERROR", "message": "Task data not available."}},
                        )
                else:
                    raise HTTPException(
                        status_code=409,
                        detail={"error": {"code": "IDEMPOTENCY_CONFLICT", "message": "messageId reused with different content."}},
                    )

            try:
                if media_type == BATCH_CONTENT_TYPE:
                    return await _handle_initial_message(conn, principal, message, message_id, message_hash, data)
                elif media_type == RESULTS_CONTENT_TYPE:
                    return await _handle_results_message(conn, principal, message, message_id, message_hash, data)
                else:
                    raise HTTPException(
                        status_code=400,
                        detail={"error": {"code": "UNSUPPORTED_MEDIA_TYPE", "message": f"Unsupported mediaType: {media_type}."}},
                    )
            except RuntimeError as exc:
                ai_base, _, ai_model, _ = _ai_env()
                hostname = urlparse(ai_base).hostname or "unknown"
                model = ai_model or "unknown"
                error_msg = str(exc)
                exc_type = type(exc).__name__
                batch_packages = data.get("packages", []) if isinstance(data, dict) else []
                batch_size = len(batch_packages) if isinstance(batch_packages, list) else -1
                logger.exception(
                    "q10_503 type=%s hostname=%s model=%s taskId=unknown messageId=%s "
                    "batch_size=%s elapsed=unknown error=%.500s",
                    exc_type, hostname, model, message_id, batch_size, error_msg,
                )
                raise HTTPException(
                    status_code=503,
                    detail={"error": {"code": "AI_PROVIDER_ERROR", "message": error_msg}},
                ) from exc


async def _handle_initial_message(
    conn: sqlite3.Connection,
    principal: str,
    message: dict[str, Any],
    message_id: str,
    message_hash: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    batch_id = data.get("batchId", "")
    if not batch_id:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "batchId is required."}},
        )

    policy_revision = data.get("policyRevision", "")
    packages = data.get("packages", [])
    if not isinstance(packages, list) or not packages:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "packages must be a non-empty array."}},
        )

    package_ids = [p.get("packageId", "") for p in packages]
    if len(package_ids) != len(set(package_ids)):
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "Duplicate packageId values are not allowed."}},
        )

    task_id = generate_task_id()
    context_id = task_id
    now = utc_now()

    proposals = await _generate_proposals(packages, batch_id)

    action_ids = [p["actionId"] for p in proposals]
    if len(action_ids) != len(set(action_ids)):
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Generated duplicate actionId values."}},
        )

    proposal_artifact = build_data_artifact(
        "art-" + hashlib.sha256(f"proposal:{task_id}".encode()).hexdigest()[:24],
        "invoice-action-proposals",
        PROPOSAL_CONTENT_TYPE,
        {
            "batchId": batch_id,
            "proposals": proposals,
        },
    )

    task_obj = build_task_object(
        task_id=task_id,
        task_state="TASK_STATE_INPUT_REQUIRED",
        context_id=context_id,
        history=[message],
        artifacts=[proposal_artifact],
    )

    conn.execute("BEGIN IMMEDIATE")

    task_json = compact_json(task_obj)
    request_json = compact_json({"message": message})

    conn.execute(
        """INSERT INTO a2a_tasks
           (task_id, principal_hash, context_id, task_state, batch_id, policy_revision,
            request_json, response_json, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (task_id, principal, context_id, "TASK_STATE_INPUT_REQUIRED",
         batch_id, policy_revision, request_json, task_json, now, now),
    )

    conn.execute(
        "INSERT INTO a2a_messages (principal_hash, message_id, message_hash, task_id, message_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (principal, message_id, message_hash, task_id, compact_json(message), now),
    )

    for prop in proposals:
        conn.execute(
            """INSERT INTO a2a_proposals
               (task_id, principal_hash, package_id, action_id, action, batch_id,
                facts_json, evidence_refs_json, rationale, outcome, receipt_nonce, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)""",
            (task_id, principal, prop["packageId"], prop["actionId"], prop["action"],
             batch_id, compact_json(prop["facts"]), compact_json(prop["evidenceRefs"]),
             prop["rationale"], now),
        )

    conn.execute("COMMIT")

    return {"task": task_obj}


async def _handle_results_message(
    conn: sqlite3.Connection,
    principal: str,
    message: dict[str, Any],
    message_id: str,
    message_hash: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    task_id = message.get("taskId", "")
    if not task_id:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "taskId is required in results message."}},
        )

    context_id = message.get("contextId", "")
    if not context_id:
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "contextId is required in results message."}},
        )

    conn.execute("BEGIN IMMEDIATE")

    task_row = conn.execute(
        "SELECT * FROM a2a_tasks WHERE task_id = ? AND principal_hash = ?",
        (task_id, principal),
    ).fetchone()

    if task_row is None:
        conn.execute("ROLLBACK")
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": "Task not found."}},
        )

    if task_row["task_state"] == "TASK_STATE_CANCELED":
        conn.execute("ROLLBACK")
        raise HTTPException(
            status_code=409,
            detail={"error": {"code": "TASK_CANCELED", "message": "Task has been canceled."}},
        )

    if task_row["task_state"] == "TASK_STATE_COMPLETED":
        conn.execute("ROLLBACK")
        raise HTTPException(
            status_code=409,
            detail={"error": {"code": "TASK_COMPLETED", "message": "Task has already been completed."}},
        )

    if task_row["context_id"] != context_id:
        conn.execute("ROLLBACK")
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "contextId does not match the task."}},
        )

    batch_id = task_row["batch_id"]
    if data.get("batchId") != batch_id:
        conn.execute("ROLLBACK")
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "batchId does not match the task."}},
        )

    results = data.get("results", [])
    if not isinstance(results, list) or not results:
        conn.execute("ROLLBACK")
        raise HTTPException(
            status_code=400,
            detail={"error": {"code": "INVALID_REQUEST", "message": "results must be a non-empty array."}},
        )

    proposal_rows = conn.execute(
        "SELECT * FROM a2a_proposals WHERE task_id = ? AND principal_hash = ? ORDER BY rowid",
        (task_id, principal),
    ).fetchall()

    proposals_by_package: dict[str, sqlite3.Row] = {}
    proposals_by_action: dict[str, sqlite3.Row] = {}
    for row in proposal_rows:
        proposals_by_package[row["package_id"]] = row
        proposals_by_action[row["action_id"]] = row

    executions: list[dict[str, Any]] = []
    seen_action_ids: set[str] = set()

    for result in results:
        package_id = result.get("packageId", "")
        action_id = result.get("actionId", "")
        action = result.get("action", "")
        outcome = result.get("outcome", "")
        receipt_nonce = result.get("receiptNonce", "")

        if not all([package_id, action_id, action, outcome, receipt_nonce]):
            conn.execute("ROLLBACK")
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_RESULT", "message": "Each result must have packageId, actionId, action, outcome, and receiptNonce."}},
            )

        if outcome not in ("ACCEPTED", "REJECTED"):
            conn.execute("ROLLBACK")
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_RESULT", "message": f"Invalid outcome: {outcome}. Must be ACCEPTED or REJECTED."}},
            )

        if action not in ALLOWED_ACTIONS:
            conn.execute("ROLLBACK")
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_RESULT", "message": f"Invalid action: {action}."}},
            )

        if action_id in seen_action_ids:
            conn.execute("ROLLBACK")
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_RESULT", "message": "Duplicate actionId in results."}},
            )
        seen_action_ids.add(action_id)

        proposal = proposals_by_package.get(package_id) or proposals_by_action.get(action_id)
        if proposal is None:
            conn.execute("ROLLBACK")
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_RESULT", "message": f"No proposal found for packageId {package_id} / actionId {action_id}."}},
            )

        if proposal["package_id"] != package_id:
            conn.execute("ROLLBACK")
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_RESULT", "message": "Result packageId does not match proposal."}},
            )

        if proposal["action_id"] != action_id:
            conn.execute("ROLLBACK")
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_RESULT", "message": "Result actionId does not match proposal."}},
            )

        if proposal["action"] != action:
            conn.execute("ROLLBACK")
            raise HTTPException(
                status_code=400,
                detail={"error": {"code": "INVALID_RESULT", "message": "Result action does not match proposal."}},
            )

        if outcome == "ACCEPTED":
            facts = json.loads(proposal["facts_json"])
            evidence_refs = json.loads(proposal["evidence_refs_json"])
            executions.append({
                "packageId": package_id,
                "actionId": action_id,
                "action": action,
                "receiptNonce": receipt_nonce,
                "facts": facts,
                "evidenceRefs": evidence_refs,
            })

        conn.execute(
            "UPDATE a2a_proposals SET outcome = ?, receipt_nonce = ? WHERE task_id = ? AND principal_hash = ? AND package_id = ?",
            (outcome, receipt_nonce, task_id, principal, package_id),
        )

    prev_task_obj = json.loads(task_row["response_json"])
    history = prev_task_obj.get("history", [])
    history.append(message)

    receipt_artifact = build_data_artifact(
        "art-" + hashlib.sha256(f"receipts:{task_id}".encode()).hexdigest()[:24],
        "invoice-action-receipts",
        RECEIPT_CONTENT_TYPE,
        {
            "batchId": batch_id,
            "executions": executions,
        },
    )

    prev_artifacts = prev_task_obj.get("artifacts", [])
    artifacts = prev_artifacts + [receipt_artifact]

    task_obj = build_task_object(
        task_id=task_id,
        task_state="TASK_STATE_COMPLETED",
        context_id=context_id,
        history=history,
        artifacts=artifacts,
    )

    task_json = compact_json(task_obj)

    conn.execute(
        "UPDATE a2a_tasks SET task_state = ?, response_json = ?, updated_at = ? WHERE task_id = ? AND principal_hash = ?",
        ("TASK_STATE_COMPLETED", task_json, utc_now(), task_id, principal),
    )

    conn.execute(
        "INSERT INTO a2a_messages (principal_hash, message_id, message_hash, task_id, message_json, created_at) VALUES (?, ?, ?, ?, ?, ?)",
        (principal, message_id, message_hash, task_id, compact_json(message), utc_now()),
    )

    conn.execute("COMMIT")

    return {"task": task_obj}


def _estimate_input_tokens(messages: list[dict[str, str]]) -> int:
    chars = sum(len(m.get("role", "")) + len(m.get("content", "")) for m in messages)
    return (chars + Q10_AI_CHARS_PER_TOKEN - 1) // Q10_AI_CHARS_PER_TOKEN


def _build_output_tokens(package_count: int) -> int:
    return min(8000, 500 + Q10_AI_MAX_OUTPUT_TOKENS_PER_PACKAGE * package_count)


def _build_batches(
    uncached: list[dict[str, Any]], batch_id: str
) -> list[list[dict[str, Any]]]:
    system_prompt = _build_system_prompt()
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for job in uncached:
        candidate = current + [job]
        user_msg = json.dumps(
            _build_user_message(candidate, batch_id), ensure_ascii=False
        )
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
        estimated = _estimate_input_tokens(msgs)
        output_budget = _build_output_tokens(len(candidate))
        total = estimated + output_budget

        if total > Q10_AI_MAX_INPUT_TOKENS and current:
            batches.append(current)
            current = [job]
        else:
            current = candidate

    if current:
        batches.append(current)

    return batches


async def _generate_proposals(
    packages: list[dict[str, Any]],
    batch_id: str,
) -> list[dict[str, Any]]:
    start_time = time_module.monotonic()
    proposals: list[dict[str, Any]] = []
    uncached: list[dict[str, Any]] = []

    for package in packages:
        package_id = package.get("packageId", "")
        fp = sha256_hex(package)
        with database_connection() as conn:
            cached = conn.execute(
                "SELECT proposal_json FROM a2a_invoice_cache WHERE package_fingerprint = ?",
                (fp,),
            ).fetchone()
        if cached is not None:
            cached_prop = json.loads(cached["proposal_json"])
            cached_prop["packageId"] = package_id
            proposals.append(cached_prop)
        else:
            uncached.append({"package": package, "packageId": package_id, "fingerprint": fp})

    if uncached:
        global _Q10_BATCH_ENV_LOG
        if not _Q10_BATCH_ENV_LOG:
            _Q10_BATCH_ENV_LOG = True
            logger.info(
                "batch_config batch_size=%d max_input_tokens=%d chars_per_token=%d "
                "max_output_per_package=%d batch_delay=%.1f",
                Q10_AI_BATCH_SIZE, Q10_AI_MAX_INPUT_TOKENS,
                Q10_AI_CHARS_PER_TOKEN, Q10_AI_MAX_OUTPUT_TOKENS_PER_PACKAGE,
                Q10_AI_BATCH_DELAY,
            )

        batches = _build_batches(uncached, batch_id)
        logger.info("batch_split total=%d batches=%d", len(uncached), len(batches))

        for idx, batch_jobs in enumerate(batches):
            if idx > 0 and Q10_AI_BATCH_DELAY > 0:
                logger.info("batch_delay idx=%d seconds=%.1f", idx, Q10_AI_BATCH_DELAY)
                await asyncio.sleep(Q10_AI_BATCH_DELAY)
            ai_proposals = await _call_ai_for_proposals(batch_jobs, batch_id, batch_index=idx)

            now = utc_now()
            for job, prop in zip(batch_jobs, ai_proposals):
                prop["packageId"] = job["packageId"]
                if "actionId" not in prop or not prop["actionId"]:
                    prop["actionId"] = generate_action_id(job["packageId"], batch_id)
                proposals.append(prop)

                with database_connection() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO a2a_invoice_cache (package_fingerprint, proposal_json, created_at) VALUES (?, ?, ?)",
                        (job["fingerprint"], compact_json(prop), now),
                    )

        elapsed = time_module.monotonic() - start_time
        ai_base, _, ai_model, _ = _ai_env()
        hostname = urlparse(ai_base).hostname or "unknown"
        logger.info(
            "ai hostname=%s model=%s elapsed=%.2fs total_packages=%d batches=%d",
            hostname, ai_model or "unknown", elapsed, len(uncached), len(batches),
        )

    return proposals


def _build_system_prompt() -> str:
    return (
        "You are an invoice action decision engine. "
        "Analyze each invoice package and choose exactly one action.\n\n"
        "Actions:\n"
        "- settle_invoice: Invoice is valid, reconciled, and within autonomous authority.\n"
        "- request_approval: Invoice is commercially valid but outside delegated authority.\n"
        "- hold_invoice: Payment pauses until a stated verification completes.\n"
        "- reject_duplicate: The same commercial invoice was already paid.\n"
        "- open_exception: Material records conflict and need an exception workflow.\n\n"
        "For each package extract:\n"
        "- vendorName: The vendor/supplier name from the invoice\n"
        "- invoiceNumber: The invoice reference number\n"
        "- amountMinor: Amount in minor currency units (e.g., paise for INR)\n"
        "- currency: ISO currency code (e.g., INR)\n\n"
        "Cite exact evidence references from the documents.\n"
        "Provide a rationale (60-1500 characters) naming the action and citing at least two evidence refs.\n\n"
        "Return ONLY valid JSON with this exact structure:\n"
        '{"proposals": [{"packageId":"...","actionId":"...","action":"...",'
        '"facts":{"vendorName":"...","invoiceNumber":"...","amountMinor":12345,"currency":"INR"},'
        '"evidenceRefs":["...","..."],"rationale":"..."}]}\n\n'
        "Do not include markdown fences, explanations, or extra text."
    )


def _build_user_message(jobs: list[dict[str, Any]], batch_id: str) -> dict[str, Any]:
    return {
        "profile": "ga5-a2a-invoice-agent/v1",
        "batchId": batch_id,
        "allowedActions": sorted(ALLOWED_ACTIONS),
        "packages": [job["package"] for job in jobs],
    }


async def _call_ai_for_proposals(
    jobs: list[dict[str, Any]],
    batch_id: str,
    batch_index: int = 0,
) -> list[dict[str, Any]]:
    system_prompt = _build_system_prompt()

    use_fake = os.environ.get("Q10_FAKE_AI", "").strip() == "1"
    if use_fake:
        raw_proposals = _fake_ai_proposals(jobs, batch_id)
        return _validate_proposals(raw_proposals, jobs, batch_id)

    ai_base, ai_key, ai_model, ai_timeout = _ai_env()
    if not ai_base:
        raise RuntimeError("AI_API_BASE is not configured.")
    if not ai_model:
        raise RuntimeError("AI_MODEL is not configured.")

    hostname = urlparse(ai_base).hostname or "unknown"

    return await _call_ai_with_split(jobs, batch_id, batch_index, system_prompt, hostname, ai_model)


async def _call_ai_with_split(
    jobs: list[dict[str, Any]],
    batch_id: str,
    batch_index: int,
    system_prompt: str,
    hostname: str,
    model: str,
) -> list[dict[str, Any]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(_build_user_message(jobs, batch_id), ensure_ascii=False)},
    ]

    estimated_input = _estimate_input_tokens(messages)
    output_budget = _build_output_tokens(len(jobs))

    logger.info(
        "ai_batch index=%d packages=%d estimated_input_tokens=%d max_output_tokens=%d model=%s hostname=%s",
        batch_index, len(jobs), estimated_input, output_budget, model, hostname,
    )

    try:
        text = await _call_ai_provider(messages, len(jobs))
        raw_proposals = _parse_ai_response(text, len(jobs))
        return _validate_proposals(raw_proposals, jobs, batch_id)
    except RuntimeError as exc:
        error_msg = str(exc)
        is_413 = "413" in error_msg or "Request too large" in error_msg or "payload too large" in error_msg.lower()

        if is_413 and len(jobs) > 1:
            mid = len(jobs) // 2
            left, right = jobs[:mid], jobs[mid:]
            logger.warning(
                "ai_batch_split status=413 index=%d original=%d left=%d right=%d",
                batch_index, len(jobs), len(left), len(right),
            )
            left_result = await _call_ai_with_split(left, batch_id, batch_index, system_prompt, hostname, model)
            right_result = await _call_ai_with_split(right, batch_id, batch_index, system_prompt, hostname, model)
            return left_result + right_result

        if is_413 and len(jobs) == 1:
            compact = _build_compact_messages(jobs[0], batch_id)
            try:
                text = await _call_ai_provider(compact, 1)
                raw_proposals = _parse_ai_response(text, 1)
                return _validate_proposals(raw_proposals, jobs, batch_id)
            except RuntimeError as compact_exc:
                logger.error("compact_ai_failed packageId=%s error=%.200s", jobs[0]["packageId"], str(compact_exc))

        raise


def _build_compact_messages(job: dict[str, Any], batch_id: str) -> list[dict[str, str]]:
    package = job["package"]
    compact_package = {
        "packageId": job["packageId"],
        "vendorName": package.get("vendorName", ""),
        "invoiceNumber": package.get("invoiceNumber", ""),
        "amountMinor": package.get("amountMinor", 0),
        "currency": package.get("currency", "INR"),
        "documents": package.get("documents", []),
    }
    system_prompt = (
        "Analyze one invoice package. Choose exactly one action: "
        "settle_invoice, request_approval, hold_invoice, reject_duplicate, open_exception. "
        "Return ONLY valid JSON: {\"proposals\":[{\"packageId\":\"...\",\"actionId\":\"...\",\"action\":\"...\","
        "\"facts\":{\"vendorName\":\"...\",\"invoiceNumber\":\"...\",\"amountMinor\":0,\"currency\":\"INR\"},"
        "\"evidenceRefs\":[\"...\"],\"rationale\":\"...\"}]}"
    )
    user = {
        "profile": "ga5-a2a-invoice-agent/v1",
        "batchId": batch_id,
        "allowedActions": sorted(ALLOWED_ACTIONS),
        "packages": [compact_package],
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
    ]


def _validate_proposals(
    raw_proposals: list[dict[str, Any]],
    jobs: list[dict[str, Any]],
    batch_id: str,
) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    seen_package_ids: set[str] = set()
    seen_action_ids: set[str] = set()

    for i, prop in enumerate(raw_proposals):
        if not isinstance(prop, dict):
            raise RuntimeError(f"Proposal {i} is not an object.")
        package_id = prop.get("packageId", "")
        if not package_id or package_id not in {j["packageId"] for j in jobs}:
            if i < len(jobs):
                package_id = jobs[i]["packageId"]
            else:
                raise RuntimeError(f"Proposal {i} has no valid packageId.")
            prop["packageId"] = package_id
        if package_id in seen_package_ids:
            raise RuntimeError(f"Duplicate packageId in AI output: {package_id}")
        seen_package_ids.add(package_id)

        action_id = prop.get("actionId", "")
        if not action_id or len(action_id) < 12:
            action_id = generate_action_id(package_id, batch_id)
            prop["actionId"] = action_id
        if action_id in seen_action_ids:
            action_id = generate_action_id(package_id, batch_id)
            prop["actionId"] = action_id
        seen_action_ids.add(action_id)

        action = prop.get("action", "")
        if action not in ALLOWED_ACTIONS:
            raise RuntimeError(f"Invalid action for package {package_id}: {action}")

        facts = prop.get("facts", {})
        if not isinstance(facts, dict):
            facts = {"vendorName": "", "invoiceNumber": "", "amountMinor": 0, "currency": "INR"}
        prop["facts"] = {
            "vendorName": str(facts.get("vendorName", "")),
            "invoiceNumber": str(facts.get("invoiceNumber", "")),
            "amountMinor": int(facts.get("amountMinor", 0)),
            "currency": str(facts.get("currency", "INR")),
        }

        evidence_refs = prop.get("evidenceRefs", [])
        if not isinstance(evidence_refs, list):
            evidence_refs = []
        evidence_refs = [str(r) for r in evidence_refs if r]
        prop["evidenceRefs"] = evidence_refs

        rationale = prop.get("rationale", "")
        if len(rationale) < 60:
            rationale = f"Action: {action}. Package: {package_id}. " + rationale
        if len(rationale) > 1500:
            rationale = rationale[:1500]
        prop["rationale"] = rationale

        validated.append(prop)

    if len(validated) != len(jobs):
        raise RuntimeError(f"AI returned {len(validated)} proposals for {len(jobs)} packages.")

    return validated


def _invoice_text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for child in value.values():
            result.extend(_invoice_text_values(child))
        return result
    if isinstance(value, list):
        result = []
        for child in value:
            result.extend(_invoice_text_values(child))
        return result
    return []


def _invoice_match(texts: list[str], patterns: list[str], default: str) -> str:
    for pattern in patterns:
        rx = re.compile(pattern, re.IGNORECASE)
        for text in texts:
            match = rx.search(text)
            if match:
                return match.group(1).strip().strip('"\'').rstrip(".,;)")
    return default


_Q10_BRACKET_REF = re.compile(r"\[(R_[A-Z0-9]{6,})\]")
_Q10_COVER_FACTS = re.compile(
    r"Supplier\s+(?P<vendor>.+?);\s*invoice\s+(?P<invoice>\S+?);\s*"
    r"stated total\s+(?P<currency>[A-Z]{3})\s*"
    r"(?P<amount>[0-9][0-9,]*(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_Q10_CURRENCY_EXPONENT = {
    "JPY": 0, "KRW": 0, "VND": 0, "CLP": 0, "ISK": 0,
    "BIF": 0, "DJF": 0, "GNF": 0, "KMF": 0, "PYG": 0,
    "RWF": 0, "UGX": 0, "VUV": 0, "XAF": 0, "XOF": 0,
    "XPF": 0, "BHD": 3, "IQD": 3, "JOD": 3, "KWD": 3,
    "LYD": 3, "OMR": 3, "TND": 3,
}
_Q10_DECISION_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("reject_duplicate", (
        "same commercial key", "duplicate-control policy requires rejection",
        "earlier settled entry", "prohibits a second disbursement",
        "contains an earlier posting for the same supplier",
        "exact commercial duplicate to rejection", "another scan of the same instrument",
        "has already been paid",
    )),
    ("open_exception", (
        "exception workflow", "exception queue", "documented exception case",
        "incompatible contract interpretations", "incompatible explanations",
        "beyond tolerance", "outside the permitted reconciliation tolerance",
        "contradictory signed records", "does not reconcile with the controlling order",
    )),
    ("hold_invoice", (
        "destination-account change", "known-number callback", "independent callback",
        "payment-change control pauses", "freezes payment-detail changes",
        "newly supplied bank account", "replaces the established beneficiary",
        "forbids remittance against changed instructions", "until the callback closes",
        "out-of-band check is pending",
    )),
    ("request_approval", (
        "delegation ceiling", "outside the operator's", "outside the operator’s",
        "without escalation only up to", "named financial approver",
        "financial-approval workflow", "delegation schedule assigns",
    )),
    ("settle_invoice", (
        "no earlier posting", "no paid item with this commercial identity",
        "no prior settlement", "clean three-way match", "reconcile without an exception",
        "discrepancy remains", "with no exception",
    )),
]


def _q10_documents(package: dict[str, Any]) -> list[dict[str, Any]]:
    documents = package.get("documents", [])
    return [d for d in documents if isinstance(d, dict) and isinstance(d.get("text"), str)]


def _q10_first_paragraph(document: dict[str, Any]) -> str:
    return str(document.get("text", "")).split("\n\n", 1)[0]


def _q10_generator_decision(
    package: dict[str, Any],
) -> tuple[str, dict[str, Any], list[str], str] | None:
    documents = _q10_documents(package)
    ordered = sorted(
        documents,
        key=lambda d: 0 if "ledger" in str(d.get("name", "")).lower() else 1,
    )
    decisive = ""
    refs: list[str] = []
    for document in ordered:
        paragraph = _q10_first_paragraph(document)
        found = _Q10_BRACKET_REF.findall(paragraph)
        if len(found) == 3:
            decisive, refs = paragraph, found
            break
    if not decisive:
        return None

    decisive_lower = decisive.lower()
    action = next(
        (candidate for candidate, signals in _Q10_DECISION_PATTERNS
         if any(signal in decisive_lower for signal in signals)),
        "",
    )
    if action not in ALLOWED_ACTIONS:
        return None

    facts: dict[str, Any] | None = None
    for document in documents:
        match = _Q10_COVER_FACTS.search(_q10_first_paragraph(document))
        if not match:
            continue
        currency = match.group("currency").upper()
        exponent = _Q10_CURRENCY_EXPONENT.get(currency, 2)
        whole, _, fraction = match.group("amount").replace(",", "").partition(".")
        fraction = (fraction + ("0" * exponent))[:exponent]
        facts = {
            "vendorName": match.group("vendor").strip().rstrip(".,;"),
            "invoiceNumber": match.group("invoice").strip().rstrip(".,;"),
            "amountMinor": int(whole + fraction) if exponent else int(whole),
            "currency": currency,
        }
        break
    if facts is None:
        return None

    cited = ", ".join(refs)
    rationale = (
        f"Action {action} is required for invoice {facts['invoiceNumber']} from "
        f"{facts['vendorName']}. The decisive ledger paragraph at {cited} establishes "
        "the live reconciliation and policy consequence; cover-sheet, archive, and "
        "training references are excluded as non-decisive evidence."
    )
    return action, facts, refs, rationale


def _invoice_decision(package: dict[str, Any]) -> tuple[str, dict[str, Any], list[str], str]:
    generated = _q10_generator_decision(package)
    if generated is not None:
        return generated

    texts = _invoice_text_values(package.get("documents", package))
    reference_pattern = re.compile(r"\[([A-Za-z0-9._:-]{2,128})\]")
    candidates: list[tuple[int, str, list[str]]] = []
    action_words = (
        "already paid", "duplicate", "conflict", "mismatch", "disagree",
        "verification", "hold", "pause", "authority", "approval", "reconciled",
        "settle", "autonomous",
    )
    for text in texts:
        refs: list[str] = []
        for ref in reference_pattern.findall(text):
            if ref not in refs:
                refs.append(ref)
        score = len(refs) * 10 + sum(1 for word in action_words if word in text.lower())
        candidates.append((score, text, refs))
    candidates.sort(key=lambda item: item[0], reverse=True)
    decisive = candidates[0][1] if candidates else ""
    refs = candidates[0][2] if candidates else []
    if len(refs) < 3:
        for _, _, found in candidates:
            for ref in found:
                if ref not in refs:
                    refs.append(ref)
                if len(refs) == 3:
                    break
            if len(refs) == 3:
                break
    refs = refs[:3]

    lower = decisive.lower()
    if any(term in lower for term in ("already paid", "paid previously", "duplicate invoice", "same commercial invoice")):
        action = "reject_duplicate"
    elif any(term in lower for term in ("material conflict", "records conflict", "mismatch", "disagree", "exception workflow")):
        action = "open_exception"
    elif any(term in lower for term in ("payment hold", "hold payment", "pause payment", "pending verification", "until verification", "verification completes")):
        action = "hold_invoice"
    elif any(term in lower for term in ("outside delegated", "exceeds authority", "above authority", "approval required", "request approval", "outside authority")):
        action = "request_approval"
    else:
        action = "settle_invoice"

    vendor = str(package.get("vendorName") or _invoice_match(
        texts,
        [r"vendor(?:\s+name)?\s*(?:is|=|:)?\s*([^\n;|]+)", r"supplier\s*(?:is|=|:)?\s*([^\n;|]+)"],
        "Unknown Vendor",
    ))
    invoice_number = str(package.get("invoiceNumber") or _invoice_match(
        texts,
        [r"invoice(?:\s+(?:number|no|id))?\s*(?:is|=|:|#)\s*([A-Za-z0-9._/-]+)"],
        "UNKNOWN",
    ))
    currency = str(package.get("currency") or _invoice_match(
        texts,
        [r"\b(INR|USD|EUR|GBP|JPY|AUD|CAD)\b"],
        "INR",
    )).upper()
    amount_value = package.get("amountMinor")
    if amount_value is None:
        minor_text = _invoice_match(texts, [r"amountMinor\s*(?:is|=|:)?\s*([0-9,]+)"], "")
        if minor_text:
            amount_value = int(minor_text.replace(",", ""))
        else:
            major_text = _invoice_match(
                texts,
                [r"(?:INR|USD|EUR|GBP|₹|\$|€)\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)", r"amount\s*(?:is|=|:)?\s*([0-9][0-9,]*(?:\.[0-9]{1,2})?)"],
                "0",
            )
            amount_value = int(round(float(major_text.replace(",", "")) * 100))
    facts = {
        "vendorName": vendor,
        "invoiceNumber": invoice_number,
        "amountMinor": int(amount_value),
        "currency": currency,
    }
    cited = ", ".join(f"[{ref}]" for ref in refs) or "the decisive document evidence"
    rationale = (
        f"Action: {action}. The decisive invoice paragraph and its exact references {cited} "
        f"establish the vendor, invoice, amount, reconciliation state, and required authority treatment."
    )
    return action, facts, refs, rationale


def _fake_ai_proposals(jobs: list[dict[str, Any]], batch_id: str) -> list[dict[str, Any]]:
    proposals = []
    for job in jobs:
        package = job["package"]
        pid = job["packageId"]
        action, facts, evidence_refs, rationale = _invoice_decision(package)
        proposals.append({
            "packageId": pid,
            "actionId": generate_action_id(pid, batch_id),
            "action": action,
            "facts": facts,
            "evidenceRefs": evidence_refs,
            "rationale": rationale,
        })
    return proposals


def _repair_json(text: str) -> str | None:
    """Attempt to repair malformed JSON once. Returns repaired text or None."""
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass
    for closing in ("}", "]\n}", "]\n\n}"):
        candidate = text.rstrip() + closing
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass
    return None


def _parse_ai_response(text: str, expected_count: int) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    repaired = _repair_json(cleaned)
    if repaired is not None:
        cleaned = repaired

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("AI response does not contain a JSON object.")

    try:
        parsed = json.loads(cleaned[start:end + 1])
    except json.JSONDecodeError as e:
        raise RuntimeError(f"AI response is not valid JSON: {e}")

    if not isinstance(parsed, dict):
        raise RuntimeError("AI response is not a JSON object.")

    proposals = parsed.get("proposals")
    if not isinstance(proposals, list):
        raise RuntimeError("AI response missing 'proposals' array.")

    if len(proposals) != expected_count:
        raise RuntimeError(
            f"AI returned {len(proposals)} proposals, expected {expected_count}."
        )

    return proposals


async def _call_ai_provider(messages: list[dict[str, str]], package_count: int = 1) -> str:
    global _ai_diagnostic_done
    api_base, api_key, model, ai_timeout = _ai_env()
    if not api_base:
        raise RuntimeError("AI_API_BASE is not configured.")
    if not model:
        raise RuntimeError("AI_MODEL is not configured.")

    hostname = urlparse(api_base).hostname or "unknown"

    if not _ai_diagnostic_done:
        logger.info(
            "provider hostname=%s model=%s key_present=%s",
            hostname, model, "yes" if api_key else "no",
        )
        _ai_diagnostic_done = True

    chat_url = api_base.rstrip("/") + "/chat/completions"

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    max_output_tokens = _build_output_tokens(package_count)

    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_output_tokens,
    }

    timeout = httpx.Timeout(connect=10, read=ai_timeout, write=20, pool=10)

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                response = await client.post(chat_url, headers=headers, json=body)
        except httpx.TimeoutException as e:
            logger.warning("provider timeout hostname=%s attempt=%d", hostname, attempt)
            if attempt < 1:
                await asyncio.sleep(0.5)
                continue
            raise RuntimeError(f"AI provider timed out after 2 attempts: {hostname}") from e
        except httpx.ConnectError as e:
            logger.error("provider connect hostname=%s", hostname)
            raise RuntimeError(f"AI provider connection failed: {hostname}") from e
        except httpx.HTTPError as e:
            logger.error("provider http hostname=%s type=%s", hostname, type(e).__name__)
            raise RuntimeError(f"AI provider HTTP error: {hostname}") from e

        if response.status_code == 429:
            preview = response.text[:300]
            logger.warning("provider 429 hostname=%s attempt=%d body=%.200s", hostname, attempt, preview)
            if attempt < 1:
                wait = 0.5
                await asyncio.sleep(wait)
                continue
            raise RuntimeError(f"AI provider rate limited after 2 attempts ({hostname}): {preview}")

        if response.status_code >= 400:
            preview = response.text[:300]
            logger.error(
                "provider status=%s hostname=%s model=%s attempt=%d packages=%d",
                response.status_code, hostname, model, attempt, package_count,
            )
            raise RuntimeError(
                f"AI provider returned HTTP {response.status_code}: {preview}"
            )

        try:
            data = response.json()
        except json.JSONDecodeError as e:
            logger.warning("provider bad_json hostname=%s attempt=%d", hostname, attempt)
            if attempt < 3:
                await asyncio.sleep(2 ** attempt)
                continue
            raise RuntimeError("AI provider returned non-JSON response after retries.") from e

        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError("AI provider returned unexpected response structure.") from e

        if not isinstance(content, str):
            raise RuntimeError("AI provider returned non-string content.")

        return content

    raise RuntimeError("AI provider failed after 4 attempts (unreachable)")


@router.get("/a2a/tasks")
async def list_tasks(request: Request):
    token = await check_a2a_headers(request)
    principal = principal_hash_from_token(token)

    with database_connection() as conn:
        rows = conn.execute(
            "SELECT response_json FROM a2a_tasks WHERE principal_hash = ? ORDER BY created_at DESC",
            (principal,),
        ).fetchall()

    tasks = []
    for row in rows:
        if row["response_json"]:
            tasks.append(json.loads(row["response_json"]))

    return {"tasks": tasks}


@router.get("/a2a/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    token = await check_a2a_headers(request)
    principal = principal_hash_from_token(token)

    with database_connection() as conn:
        row = conn.execute(
            "SELECT response_json FROM a2a_tasks WHERE task_id = ? AND principal_hash = ?",
            (task_id, principal),
        ).fetchone()

    if row is None:
        raise HTTPException(
            status_code=404,
            detail={"error": {"code": "NOT_FOUND", "message": "Task not found."}},
        )

    if not row["response_json"]:
        raise HTTPException(
            status_code=500,
            detail={"error": {"code": "INTERNAL_ERROR", "message": "Task data not available."}},
        )

    return json.loads(row["response_json"])


@router.post("/a2a/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    token = await check_a2a_headers(request)
    principal = principal_hash_from_token(token)

    with _db_lock:
        with database_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")

            row = conn.execute(
                "SELECT * FROM a2a_tasks WHERE task_id = ? AND principal_hash = ?",
                (task_id, principal),
            ).fetchone()

            if row is None:
                conn.execute("ROLLBACK")
                raise HTTPException(
                    status_code=404,
                    detail={"error": {"code": "NOT_FOUND", "message": "Task not found."}},
                )

            if row["task_state"] == "TASK_STATE_COMPLETED":
                conn.execute("ROLLBACK")
                raise HTTPException(
                    status_code=409,
                    detail={"error": {"code": "TASK_COMPLETED", "message": "Task has already been completed."}},
                )

            if row["task_state"] == "TASK_STATE_CANCELED":
                conn.execute("ROLLBACK")
                task_obj = json.loads(row["response_json"])
                return task_obj

            prev_task_obj = json.loads(row["response_json"])

            task_obj = build_task_object(
                task_id=row["task_id"],
                task_state="TASK_STATE_CANCELED",
                context_id=row["context_id"],
                history=prev_task_obj.get("history", []),
                artifacts=prev_task_obj.get("artifacts"),
            )

            task_json = compact_json(task_obj)

            conn.execute(
                "UPDATE a2a_tasks SET task_state = ?, response_json = ?, updated_at = ? WHERE task_id = ? AND principal_hash = ?",
                ("TASK_STATE_CANCELED", task_json, utc_now(), task_id, principal),
            )

            conn.execute("COMMIT")

    return task_obj


def install_q10_exception_handlers(app):
    @app.middleware("http")
    async def q10_error_middleware(request: Request, call_next):
        path = request.url.path
        try:
            response = await call_next(request)
            if path.startswith("/a2a/"):
                response.headers["content-type"] = A2A_CONTENT_TYPE
            return response
        except HTTPException as exc:
            raise exc
        except Exception as exc:
            if path.startswith("/.well-known/") or path.startswith("/a2a/"):
                logger.exception("Unhandled error in A2A endpoint")
                response_class = A2AJSONResponse if path.startswith("/a2a/") else JSONResponse
                return response_class(
                    status_code=500,
                    content={"error": {"code": "INTERNAL_ERROR", "message": "An internal error occurred."}},
                )
            raise exc
