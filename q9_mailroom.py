from __future__ import annotations

import base64
import binascii
import hashlib
import httpx
import json
import logging
import os
import re
import sqlite3
import threading
import time as time_module
from urllib.parse import urlparse

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PublicKey,
)
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from q9_decision_engine import (
    AIOutputError,
    AIProviderError,
    ProposalValidationError,
    generate_validated_proposals,
    validate_proposal,
)


logger = logging.getLogger(__name__)

if not logging.root.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )


def safe_validation_errors(
    error: ValidationError,
) -> list[dict[str, Any]]:
    """
    Convert Pydantic validation errors to a JSON-safe structure.
    """
    safe: list[dict[str, Any]] = []

    for err in error.errors():
        entry: dict[str, Any] = {
            "type": err.get("type", "unknown"),
            "loc": list(err.get("loc", [])),
            "msg": err.get("msg", ""),
            "input": str(err.get("input", "")),
        }

        ctx = err.get("ctx")

        if ctx is not None:
            cleaned_ctx: dict[str, Any] = {}

            for key, value in ctx.items():
                try:
                    json.dumps(value)
                    cleaned_ctx[key] = value
                except TypeError:
                    cleaned_ctx[key] = str(value)

            entry["ctx"] = cleaned_ctx

        safe.append(entry)

    return safe


router = APIRouter()

PROFILE = "ga5-mailroom-action-gate/v2"

DATABASE_PATH = Path("mailroom_agent.db")

ALLOWED_ACTIONS = {
    "create_draft",
    "update_internal_record",
    "send_approved_notice",
    "request_confirmation",
    "quarantine_item",
    "no_action",
}

CALL_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9._:-]{12,128}$"
)

_db_lock = threading.RLock()


# ============================================================
# Canonical JSON and hashes
# ============================================================

def canonical_json_bytes(value: Any) -> bytes:
    """
    Encode JSON using:
    - recursively sorted object keys
    - compact separators
    - original array order
    - UTF-8
    - normal JSON primitive spellings
    """

    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )

    return text.encode("utf-8")


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(
        canonical_json_bytes(value)
    ).hexdigest()


def calculate_input_digest(
    dossiers: list[dict[str, Any]],
) -> str:
    """
    The assignment requires inputDigest to cover only the canonical
    JSON representation of the dossiers array.
    """

    return sha256_hex(dossiers)


def calculate_dossier_fingerprint(
    dossier: dict[str, Any],
) -> str:
    """
    Cache decisions using canonical dossier content rather than
    evaluationId.
    """

    return sha256_hex(dossier)


def create_stable_call_id(
    dossier_id: str,
    dossier_fingerprint: str,
) -> str:
    """
    Stable across later evaluations when dossier content is unchanged.

    Format contains only allowed callId characters.
    """

    digest = hashlib.sha256(
        (
            f"{dossier_id}:"
            f"{dossier_fingerprint}:"
            f"{PROFILE}"
        ).encode("utf-8")
    ).hexdigest()

    call_id = f"mailroom:{digest[:32]}"

    if not CALL_ID_PATTERN.fullmatch(call_id):
        raise RuntimeError(
            "Generated callId does not satisfy the contract."
        )

    return call_id


def calculate_proposal_digest(
    proposal: dict[str, Any],
) -> str:
    """
    The proposal digest includes exactly:
    - dossierId
    - callId
    - action
    - target
    - payload
    - evidence sorted alphabetically
    """

    normalized = {
        "dossierId": proposal["dossierId"],
        "callId": proposal["callId"],
        "action": proposal["action"],
        "target": proposal.get("target"),
        "payload": proposal["payload"],
        "evidence": sorted(
            proposal["evidence"]
        ),
    }

    return sha256_hex(normalized)


# ============================================================
# Pydantic request models
# ============================================================

class StrictModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )


class PublicKeyJwk(StrictModel):
    kty: Literal["OKP"]
    crv: Literal["Ed25519"]
    x: str = Field(min_length=1, max_length=256)


class ReceiptVerifier(StrictModel):
    algorithm: Literal["Ed25519"]
    publicKeyJwk: PublicKeyJwk


class Corpus(StrictModel):
    coreId: str = Field(min_length=1, max_length=512)
    auditId: str = Field(min_length=1, max_length=512)
    stableCount: int = Field(ge=0, le=1000)
    freshCount: int = Field(ge=0, le=1000)


class SourceLine(StrictModel):
    lineId: str = Field(min_length=1, max_length=512)
    text: str = Field(max_length=100_000)


class DossierSource(StrictModel):
    sourceId: str = Field(min_length=1, max_length=512)
    kind: str = Field(min_length=1, max_length=512)
    provenance: str = Field(min_length=1, max_length=2048)
    title: str = Field(max_length=5000)
    lines: list[SourceLine] = Field(
        min_length=1,
        max_length=20_000,
    )

    @field_validator("lines")
    @classmethod
    def unique_line_ids(
        cls,
        lines: list[SourceLine],
    ) -> list[SourceLine]:
        line_ids = [
            line.lineId
            for line in lines
        ]

        if len(line_ids) != len(set(line_ids)):
            raise ValueError(
                "Duplicate lineId values are not allowed "
                "inside a source."
            )

        return lines


class Dossier(StrictModel):
    dossierId: str = Field(min_length=1, max_length=512)
    partition: Literal[
        "stable_core",
        "fresh_audit",
    ]
    receivedAt: str = Field(min_length=1, max_length=128)
    mailbox: str = Field(min_length=1, max_length=2048)
    objective: str = Field(min_length=1, max_length=20_000)
    sources: list[DossierSource] = Field(
        min_length=1,
        max_length=10_000,
    )

    @field_validator("receivedAt")
    @classmethod
    def validate_received_at(
        cls,
        value: str,
    ) -> str:
        candidate = value

        if candidate.endswith("Z"):
            candidate = (
                candidate[:-1]
                + "+00:00"
            )

        try:
            datetime.fromisoformat(candidate)
        except ValueError as error:
            raise ValueError(
                "receivedAt must be an ISO timestamp."
            ) from error

        return value

    @field_validator("sources")
    @classmethod
    def unique_source_and_line_ids(
        cls,
        sources: list[DossierSource],
    ) -> list[DossierSource]:
        source_ids = [
            source.sourceId
            for source in sources
        ]

        if len(source_ids) != len(set(source_ids)):
            raise ValueError(
                "Duplicate sourceId values are not allowed."
            )

        all_line_ids: list[str] = []

        for source in sources:
            all_line_ids.extend(
                line.lineId
                for line in source.lines
            )

        if len(all_line_ids) != len(set(all_line_ids)):
            raise ValueError(
                "Every lineId must be unique within a dossier."
            )

        return sources


class ProposeRequest(StrictModel):
    profile: Literal[
        "ga5-mailroom-action-gate/v2"
    ]
    operation: Literal["propose"]
    evaluationId: str = Field(
        min_length=1,
        max_length=512,
    )
    receiptVerifier: ReceiptVerifier
    corpus: Corpus
    allowedActions: list[str] = Field(
        min_length=1,
        max_length=100,
    )
    dossiers: list[Dossier] = Field(
        min_length=1,
        max_length=200,
    )

    @field_validator("allowedActions")
    @classmethod
    def validate_allowed_actions(
        cls,
        actions: list[str],
    ) -> list[str]:
        if len(actions) != len(set(actions)):
            raise ValueError(
                "allowedActions contains duplicates."
            )

        unknown = set(actions) - ALLOWED_ACTIONS

        if unknown:
            raise ValueError(
                f"Unknown allowed actions: {sorted(unknown)}"
            )

        if set(actions) != ALLOWED_ACTIONS:
            raise ValueError(
                "allowedActions must contain exactly the six "
                "supported actions."
            )

        return actions

    @field_validator("dossiers")
    @classmethod
    def unique_dossier_ids(
        cls,
        dossiers: list[Dossier],
    ) -> list[Dossier]:
        dossier_ids = [
            dossier.dossierId
            for dossier in dossiers
        ]

        if len(dossier_ids) != len(set(dossier_ids)):
            raise ValueError(
                "Duplicate dossierId values are not allowed."
            )

        return dossiers


class CommitReceipt(StrictModel):
    dossierId: str = Field(min_length=1, max_length=512)
    callId: str = Field(min_length=12, max_length=128)
    action: str = Field(min_length=1, max_length=128)
    accepted: bool
    proposalDigest: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    receiptId: str = Field(min_length=1, max_length=1024)
    receiptSignature: str = Field(
        min_length=1,
        max_length=4096,
    )

    @field_validator("callId")
    @classmethod
    def validate_call_id(
        cls,
        value: str,
    ) -> str:
        if not CALL_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "callId contains invalid characters."
            )

        return value

    @field_validator("action")
    @classmethod
    def validate_action(
        cls,
        value: str,
    ) -> str:
        if value not in ALLOWED_ACTIONS:
            raise ValueError(
                "Receipt contains an unknown action."
            )

        return value


class CommitRequest(StrictModel):
    profile: Literal[
        "ga5-mailroom-action-gate/v2"
    ]
    operation: Literal["commit"]
    evaluationId: str = Field(
        min_length=1,
        max_length=512,
    )
    inputDigest: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    receipts: list[CommitReceipt] = Field(
        min_length=1,
        max_length=200,
    )

    @field_validator("receipts")
    @classmethod
    def unique_receipts(
        cls,
        receipts: list[CommitReceipt],
    ) -> list[CommitReceipt]:
        receipt_ids = [
            receipt.receiptId
            for receipt in receipts
        ]

        if len(receipt_ids) != len(set(receipt_ids)):
            raise ValueError(
                "Duplicate receiptId values are not allowed."
            )

        call_ids = [
            receipt.callId
            for receipt in receipts
        ]

        if len(call_ids) != len(set(call_ids)):
            raise ValueError(
                "Duplicate callId values are not allowed."
            )

        dossier_ids = [
            receipt.dossierId
            for receipt in receipts
        ]

        if len(dossier_ids) != len(set(dossier_ids)):
            raise ValueError(
                "Duplicate dossierId values are not allowed."
            )

        return receipts


# ============================================================
# Database
# ============================================================

@contextmanager
def database_connection() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
        isolation_level=None,
        check_same_thread=False,
    )

    connection.row_factory = sqlite3.Row

    try:
        connection.execute(
            "PRAGMA foreign_keys = ON"
        )
        connection.execute(
            "PRAGMA journal_mode = WAL"
        )
        connection.execute(
            "PRAGMA synchronous = FULL"
        )

        yield connection
    finally:
        connection.close()


def initialise_database() -> None:
    with _db_lock:
        with database_connection() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluations (
                    evaluation_id TEXT PRIMARY KEY,
                    profile TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    request_fingerprint TEXT NOT NULL,
                    input_digest TEXT NOT NULL,
                    verifier_json TEXT NOT NULL,
                    corpus_json TEXT NOT NULL,
                    allowed_actions_json TEXT NOT NULL,
                    response_json TEXT,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluation_dossiers (
                    evaluation_id TEXT NOT NULL,
                    dossier_id TEXT NOT NULL,
                    dossier_fingerprint TEXT NOT NULL,
                    dossier_json TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    position INTEGER NOT NULL,

                    PRIMARY KEY (
                        evaluation_id,
                        dossier_id
                    ),

                    UNIQUE (
                        evaluation_id,
                        call_id
                    ),

                    FOREIGN KEY (
                        evaluation_id
                    )
                    REFERENCES evaluations (
                        evaluation_id
                    )
                    ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS dossier_cache (
                    dossier_id TEXT NOT NULL,
                    dossier_fingerprint TEXT NOT NULL,
                    dossier_json TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    proposal_json TEXT,
                    proposal_digest TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    PRIMARY KEY (
                        dossier_id,
                        dossier_fingerprint
                    ),

                    UNIQUE (
                        call_id
                    )
                );

                CREATE TABLE IF NOT EXISTS commits (
                    evaluation_id TEXT PRIMARY KEY,
                    request_fingerprint TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    response_json TEXT,
                    state TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    FOREIGN KEY (
                        evaluation_id
                    )
                    REFERENCES evaluations (
                        evaluation_id
                    )
                    ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS receipts (
                    evaluation_id TEXT NOT NULL,
                    receipt_id TEXT NOT NULL,
                    dossier_id TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    accepted INTEGER NOT NULL,
                    proposal_digest TEXT NOT NULL,
                    receipt_signature TEXT NOT NULL,
                    receipt_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,

                    PRIMARY KEY (
                        evaluation_id,
                        receipt_id
                    ),

                    UNIQUE (
                        evaluation_id,
                        call_id
                    ),

                    FOREIGN KEY (
                        evaluation_id
                    )
                    REFERENCES evaluations (
                        evaluation_id
                    )
                    ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS action_effects (
                    evaluation_id TEXT NOT NULL,
                    dossier_id TEXT NOT NULL,
                    call_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    proposal_digest TEXT NOT NULL,
                    receipt_id TEXT NOT NULL,
                    effect_status TEXT NOT NULL,
                    effect_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,

                    PRIMARY KEY (
                        evaluation_id,
                        call_id
                    ),

                    UNIQUE (
                        evaluation_id,
                        receipt_id
                    ),

                    FOREIGN KEY (
                        evaluation_id
                    )
                    REFERENCES evaluations (
                        evaluation_id
                    )
                    ON DELETE CASCADE
                );
                """
            )


initialise_database()


# ============================================================
# Persistence helpers
# ============================================================

def utc_now() -> str:
    return datetime.utcnow().isoformat(
        timespec="microseconds"
    ) + "Z"


def compact_json_text(
    value: Any,
) -> str:
    return canonical_json_bytes(
        value
    ).decode("utf-8")


def load_evaluation(
    evaluation_id: str,
) -> sqlite3.Row | None:
    with database_connection() as connection:
        return connection.execute(
            """
            SELECT *
            FROM evaluations
            WHERE evaluation_id = ?
            """,
            (evaluation_id,),
        ).fetchone()


def save_new_evaluation(
    request: ProposeRequest,
    raw_request: dict[str, Any],
) -> dict[str, Any]:
    dossiers = [
        dossier.model_dump(
            mode="json"
        )
        for dossier in request.dossiers
    ]

    input_digest = calculate_input_digest(
        dossiers
    )

    request_fingerprint = sha256_hex(
        raw_request
    )

    now = utc_now()

    with _db_lock:
        with database_connection() as connection:
            try:
                connection.execute("BEGIN IMMEDIATE")

                existing = connection.execute(
                    """
                    SELECT *
                    FROM evaluations
                    WHERE evaluation_id = ?
                    """,
                    (request.evaluationId,),
                ).fetchone()

                if existing is not None:
                    connection.execute("ROLLBACK")

                    if (
                        existing["request_fingerprint"]
                        == request_fingerprint
                    ):
                        return {
                            "kind": "exact_replay",
                            "evaluation": dict(existing),
                        }

                    raise HTTPException(
                        status_code=409,
                        detail=(
                            "evaluationId already exists with "
                            "different propose content."
                        ),
                    )

                connection.execute(
                    """
                    INSERT INTO evaluations (
                        evaluation_id,
                        profile,
                        operation,
                        request_fingerprint,
                        input_digest,
                        verifier_json,
                        corpus_json,
                        allowed_actions_json,
                        response_json,
                        state,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.evaluationId,
                        request.profile,
                        request.operation,
                        request_fingerprint,
                        input_digest,
                        compact_json_text(
                            request.receiptVerifier.model_dump(
                                mode="json"
                            )
                        ),
                        compact_json_text(
                            request.corpus.model_dump(
                                mode="json"
                            )
                        ),
                        compact_json_text(
                            request.allowedActions
                        ),
                        None,
                        "validated",
                        now,
                        now,
                    ),
                )

                generated_call_ids: set[str] = set()

                for position, dossier in enumerate(
                    dossiers
                ):
                    dossier_id = dossier["dossierId"]

                    dossier_fingerprint = (
                        calculate_dossier_fingerprint(
                            dossier
                        )
                    )

                    call_id = create_stable_call_id(
                        dossier_id,
                        dossier_fingerprint,
                    )

                    if call_id in generated_call_ids:
                        raise RuntimeError(
                            "Generated duplicate callId."
                        )

                    generated_call_ids.add(call_id)

                    connection.execute(
                        """
                        INSERT INTO evaluation_dossiers (
                            evaluation_id,
                            dossier_id,
                            dossier_fingerprint,
                            dossier_json,
                            call_id,
                            position
                        )
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            request.evaluationId,
                            dossier_id,
                            dossier_fingerprint,
                            compact_json_text(dossier),
                            call_id,
                            position,
                        ),
                    )

                    connection.execute(
                        """
                        INSERT INTO dossier_cache (
                            dossier_id,
                            dossier_fingerprint,
                            dossier_json,
                            call_id,
                            proposal_json,
                            proposal_digest,
                            created_at,
                            updated_at
                        )
                        VALUES (?, ?, ?, ?, NULL, NULL, ?, ?)
                        ON CONFLICT (
                            dossier_id,
                            dossier_fingerprint
                        )
                        DO UPDATE SET
                            dossier_json = excluded.dossier_json,
                            updated_at = excluded.updated_at
                        """,
                        (
                            dossier_id,
                            dossier_fingerprint,
                            compact_json_text(dossier),
                            call_id,
                            now,
                            now,
                        ),
                    )

                connection.execute("COMMIT")

            except HTTPException:
                raise

            except Exception:
                try:
                    connection.execute("ROLLBACK")
                except sqlite3.Error:
                    pass

                raise

    return {
        "kind": "new",
        "evaluationId": request.evaluationId,
        "inputDigest": input_digest,
        "requestFingerprint": request_fingerprint,
        "dossierCount": len(dossiers),
    }


# ============================================================
# Layer 2 persistence functions
# ============================================================

def get_evaluation_dossier_rows(
    evaluation_id: str,
) -> list[sqlite3.Row]:
    with database_connection() as connection:
        return connection.execute(
            """
            SELECT
                ed.evaluation_id,
                ed.dossier_id,
                ed.dossier_fingerprint,
                ed.dossier_json,
                ed.call_id,
                ed.position,
                dc.proposal_json,
                dc.proposal_digest
            FROM evaluation_dossiers AS ed
            LEFT JOIN dossier_cache AS dc
              ON dc.dossier_id = ed.dossier_id
             AND dc.dossier_fingerprint =
                 ed.dossier_fingerprint
            WHERE ed.evaluation_id = ?
            ORDER BY ed.position ASC
            """,
            (evaluation_id,),
        ).fetchall()


def build_and_persist_propose_response(
    request: ProposeRequest,
) -> dict[str, Any]:
    """
    Reuses cached decisions whenever the dossier ID and canonical
    dossier fingerprint are unchanged.

    Only uncached dossiers are sent to the model.
    """

    evaluation = load_evaluation(
        request.evaluationId
    )

    if evaluation is None:
        raise RuntimeError(
            "Evaluation disappeared before proposal generation."
        )

    rows = get_evaluation_dossier_rows(
        request.evaluationId
    )

    if len(rows) != len(request.dossiers):
        raise RuntimeError(
            "Persisted dossier count does not match request."
        )

    dossiers_by_id = {
        dossier.dossierId: dossier.model_dump(
            mode="json"
        )
        for dossier in request.dossiers
    }

    cached_proposals: dict[
        str,
        dict[str, Any],
    ] = {}

    uncached_jobs: list[
        dict[str, Any]
    ] = []

    for row in rows:
        dossier_id = row["dossier_id"]
        dossier = dossiers_by_id[
            dossier_id
        ]

        if row["proposal_json"]:
            cached = json.loads(
                row["proposal_json"]
            )

            # Revalidate cached content before trusting it.
            validated_cached = validate_proposal(
                proposal=cached,
                dossier=dossier,
                expected_call_id=row["call_id"],
                request_allowed_actions=set(
                    request.allowedActions
                ),
            )

            cached_digest = (
                calculate_proposal_digest(
                    validated_cached
                )
            )

            if (
                row["proposal_digest"]
                != cached_digest
            ):
                raise RuntimeError(
                    "Cached proposal digest mismatch."
                )

            cached_proposals[
                dossier_id
            ] = validated_cached

        else:
            uncached_jobs.append(
                {
                    "dossier": dossier,
                    "callId": row["call_id"],
                    "dossierFingerprint": (
                        row[
                            "dossier_fingerprint"
                        ]
                    ),
                }
            )

    generated_proposals = (
        generate_validated_proposals(
            dossier_jobs=uncached_jobs,
            allowed_actions=(
                request.allowedActions
            ),
        )
    )

    generated_by_id = {
        proposal["dossierId"]: proposal
        for proposal in generated_proposals
    }

    now = utc_now()

    with _db_lock:
        with database_connection() as connection:
            try:
                connection.execute(
                    "BEGIN IMMEDIATE"
                )

                for job in uncached_jobs:
                    dossier = job["dossier"]
                    dossier_id = dossier[
                        "dossierId"
                    ]

                    proposal = (
                        generated_by_id[
                            dossier_id
                        ]
                    )

                    proposal_digest = (
                        calculate_proposal_digest(
                            proposal
                        )
                    )

                    updated = connection.execute(
                        """
                        UPDATE dossier_cache
                        SET
                            proposal_json = ?,
                            proposal_digest = ?,
                            updated_at = ?
                        WHERE dossier_id = ?
                          AND dossier_fingerprint = ?
                          AND proposal_json IS NULL
                        """,
                        (
                            compact_json_text(
                                proposal
                            ),
                            proposal_digest,
                            now,
                            dossier_id,
                            job[
                                "dossierFingerprint"
                            ],
                        ),
                    )

                    if updated.rowcount != 1:
                        # Another concurrent request may have completed
                        # the same dossier first. Read and verify it.
                        concurrent = connection.execute(
                            """
                            SELECT
                                proposal_json,
                                proposal_digest
                            FROM dossier_cache
                            WHERE dossier_id = ?
                              AND dossier_fingerprint = ?
                            """,
                            (
                                dossier_id,
                                job[
                                    "dossierFingerprint"
                                ],
                            ),
                        ).fetchone()

                        if (
                            concurrent is None
                            or not concurrent[
                                "proposal_json"
                            ]
                        ):
                            raise RuntimeError(
                                "Could not persist proposal."
                            )

                        concurrent_proposal = (
                            json.loads(
                                concurrent[
                                    "proposal_json"
                                ]
                            )
                        )

                        validated_concurrent = (
                            validate_proposal(
                                proposal=(
                                    concurrent_proposal
                                ),
                                dossier=dossier,
                                expected_call_id=(
                                    job["callId"]
                                ),
                                request_allowed_actions=(
                                    set(
                                        request.allowedActions
                                    )
                                ),
                            )
                        )

                        concurrent_digest = (
                            calculate_proposal_digest(
                                validated_concurrent
                            )
                        )

                        if (
                            concurrent_digest
                            != concurrent[
                                "proposal_digest"
                            ]
                        ):
                            raise RuntimeError(
                                "Concurrent cached proposal "
                                "digest mismatch."
                            )

                        generated_by_id[
                            dossier_id
                        ] = (
                            validated_concurrent
                        )

                ordered_proposals: list[
                    dict[str, Any]
                ] = []

                seen_call_ids: set[str] = set()

                for row in rows:
                    dossier_id = row[
                        "dossier_id"
                    ]

                    proposal = (
                        cached_proposals.get(
                            dossier_id
                        )
                        or generated_by_id.get(
                            dossier_id
                        )
                    )

                    if proposal is None:
                        # It may have been written by a concurrent
                        # request after our first read.
                        cache_row = (
                            connection.execute(
                                """
                                SELECT proposal_json
                                FROM dossier_cache
                                WHERE dossier_id = ?
                                  AND dossier_fingerprint = ?
                                """,
                                (
                                    dossier_id,
                                    row[
                                        "dossier_fingerprint"
                                    ],
                                ),
                            ).fetchone()
                        )

                        if (
                            cache_row is None
                            or not cache_row[
                                "proposal_json"
                            ]
                        ):
                            raise RuntimeError(
                                "Missing final proposal."
                            )

                        proposal = json.loads(
                            cache_row[
                                "proposal_json"
                            ]
                        )

                    if (
                        proposal["callId"]
                        in seen_call_ids
                    ):
                        raise RuntimeError(
                            "Duplicate callId in final response."
                        )

                    seen_call_ids.add(
                        proposal["callId"]
                    )

                    ordered_proposals.append(
                        proposal
                    )

                response = {
                    "profile": PROFILE,
                    "evaluationId": (
                        request.evaluationId
                    ),
                    "status": (
                        "awaiting_receipts"
                    ),
                    "inputDigest": evaluation[
                        "input_digest"
                    ],
                    "proposals": (
                        ordered_proposals
                    ),
                }

                response_json = compact_json_text(
                    response
                )

                connection.execute(
                    """
                    UPDATE evaluations
                    SET
                        response_json = ?,
                        state = ?,
                        updated_at = ?
                    WHERE evaluation_id = ?
                    """,
                    (
                        response_json,
                        "awaiting_receipts",
                        now,
                        request.evaluationId,
                    ),
                )

                connection.execute(
                    "COMMIT"
                )

            except Exception:
                try:
                    connection.execute(
                        "ROLLBACK"
                    )
                except sqlite3.Error:
                    pass

                raise

    return response


def load_persisted_propose_response(
    evaluation_id: str,
) -> dict[str, Any] | None:
    evaluation = load_evaluation(
        evaluation_id
    )

    if (
        evaluation is None
        or not evaluation["response_json"]
    ):
        return None

    return json.loads(
        evaluation["response_json"]
    )


# ============================================================
# Layer 3: receipt verification and commit lifecycle
# ============================================================

class ReceiptVerificationError(ValueError):
    """Raised when a receipt cannot be safely accepted."""


def decode_base64url_no_padding(
    value: str,
) -> bytes:
    """
    Decode a base64url JWK value.

    Ed25519 public key JWK `x` values normally omit padding.
    """

    if not isinstance(value, str) or not value:
        raise ReceiptVerificationError(
            "JWK x must be a non-empty string."
        )

    padding = "=" * (-len(value) % 4)

    try:
        return base64.urlsafe_b64decode(
            value + padding
        )
    except (
        ValueError,
        binascii.Error,
    ) as error:
        raise ReceiptVerificationError(
            "JWK x is not valid base64url."
        ) from error


def decode_receipt_signature(
    value: str,
) -> bytes:
    """
    receiptSignature uses ordinary base64 according to the assignment.
    """

    if not isinstance(value, str) or not value:
        raise ReceiptVerificationError(
            "receiptSignature is missing."
        )

    try:
        signature = base64.b64decode(
            value,
            validate=True,
        )
    except (
        ValueError,
        binascii.Error,
    ) as error:
        raise ReceiptVerificationError(
            "receiptSignature is not valid base64."
        ) from error

    if len(signature) != 64:
        raise ReceiptVerificationError(
            "Ed25519 receiptSignature must be 64 bytes."
        )

    return signature


def import_receipt_verifier(
    verifier_json: str,
) -> Ed25519PublicKey:
    try:
        verifier = json.loads(
            verifier_json
        )
    except json.JSONDecodeError as error:
        raise ReceiptVerificationError(
            "Persisted receipt verifier is malformed."
        ) from error

    if not isinstance(verifier, dict):
        raise ReceiptVerificationError(
            "Persisted receipt verifier must be an object."
        )

    if set(verifier) != {
        "algorithm",
        "publicKeyJwk",
    }:
        raise ReceiptVerificationError(
            "Receipt verifier contains incorrect fields."
        )

    if verifier["algorithm"] != "Ed25519":
        raise ReceiptVerificationError(
            "Unsupported receipt verifier algorithm."
        )

    jwk = verifier["publicKeyJwk"]

    if not isinstance(jwk, dict):
        raise ReceiptVerificationError(
            "publicKeyJwk must be an object."
        )

    if set(jwk) != {
        "kty",
        "crv",
        "x",
    }:
        raise ReceiptVerificationError(
            "publicKeyJwk contains incorrect fields."
        )

    if jwk["kty"] != "OKP":
        raise ReceiptVerificationError(
            "publicKeyJwk.kty must be OKP."
        )

    if jwk["crv"] != "Ed25519":
        raise ReceiptVerificationError(
            "publicKeyJwk.crv must be Ed25519."
        )

    public_key_bytes = decode_base64url_no_padding(
        jwk["x"]
    )

    if len(public_key_bytes) != 32:
        raise ReceiptVerificationError(
            "Ed25519 public key must be 32 bytes."
        )

    try:
        return Ed25519PublicKey.from_public_bytes(
            public_key_bytes
        )
    except ValueError as error:
        raise ReceiptVerificationError(
            "Invalid Ed25519 public key."
        ) from error


def normalised_receipt_without_signature(
    receipt: CommitReceipt,
) -> dict[str, Any]:
    """
    Keep every receipt field except receiptSignature.
    """

    return {
        "dossierId": receipt.dossierId,
        "callId": receipt.callId,
        "action": receipt.action,
        "accepted": receipt.accepted,
        "proposalDigest": receipt.proposalDigest,
        "receiptId": receipt.receiptId,
    }


def receipt_signature_message(
    evaluation_id: str,
    input_digest: str,
    receipt: CommitReceipt,
) -> bytes:
    """
    Build the exact recursively key-sorted compact JSON signed by the
    grader.
    """

    message = {
        "profile": PROFILE,
        "evaluationId": evaluation_id,
        "inputDigest": input_digest,
        "receipt": (
            normalised_receipt_without_signature(
                receipt
            )
        ),
    }

    return canonical_json_bytes(message)


def verify_receipt_signature(
    public_key: Ed25519PublicKey,
    evaluation_id: str,
    input_digest: str,
    receipt: CommitReceipt,
) -> None:
    signature = decode_receipt_signature(
        receipt.receiptSignature
    )

    message = receipt_signature_message(
        evaluation_id=evaluation_id,
        input_digest=input_digest,
        receipt=receipt,
    )

    try:
        public_key.verify(
            signature,
            message,
        )
    except InvalidSignature as error:
        raise ReceiptVerificationError(
            f"Invalid receipt signature for "
            f"callId {receipt.callId!r}."
        ) from error


def get_persisted_proposals_for_commit(
    evaluation_id: str,
) -> list[dict[str, Any]]:
    """
    Load every persisted proposal in original dossier order.
    """

    rows = get_evaluation_dossier_rows(
        evaluation_id
    )

    proposals: list[dict[str, Any]] = []

    for row in rows:
        if not row["proposal_json"]:
            raise ReceiptVerificationError(
                "Evaluation does not have complete proposals."
            )

        proposal = json.loads(
            row["proposal_json"]
        )

        calculated_digest = (
            calculate_proposal_digest(
                proposal
            )
        )

        if (
            calculated_digest
            != row["proposal_digest"]
        ):
            raise ReceiptVerificationError(
                "Persisted proposal digest mismatch."
            )

        proposals.append(
            {
                "proposal": proposal,
                "proposalDigest": (
                    calculated_digest
                ),
                "position": row["position"],
            }
        )

    return proposals


def validate_receipts_against_proposals(
    request: CommitRequest,
    proposal_records: list[dict[str, Any]],
    public_key: Ed25519PublicKey,
) -> list[dict[str, Any]]:
    """
    Validate everything before any database write or effect.

    A commit is all-or-nothing.
    """

    if (
        len(request.receipts)
        != len(proposal_records)
    ):
        raise ReceiptVerificationError(
            "Commit must contain exactly one receipt "
            "for every proposal."
        )

    proposals_by_call_id: dict[
        str,
        dict[str, Any],
    ] = {}

    proposals_by_dossier_id: dict[
        str,
        dict[str, Any],
    ] = {}

    for record in proposal_records:
        proposal = record["proposal"]
        call_id = proposal["callId"]
        dossier_id = proposal["dossierId"]

        if call_id in proposals_by_call_id:
            raise ReceiptVerificationError(
                "Persisted proposals contain duplicate callIds."
            )

        if dossier_id in proposals_by_dossier_id:
            raise ReceiptVerificationError(
                "Persisted proposals contain duplicate dossierIds."
            )

        proposals_by_call_id[
            call_id
        ] = record

        proposals_by_dossier_id[
            dossier_id
        ] = record

    verified: list[dict[str, Any]] = []

    seen_receipt_ids: set[str] = set()
    seen_call_ids: set[str] = set()
    seen_dossier_ids: set[str] = set()

    for receipt in request.receipts:
        if receipt.receiptId in seen_receipt_ids:
            raise ReceiptVerificationError(
                "Duplicate receiptId."
            )

        if receipt.callId in seen_call_ids:
            raise ReceiptVerificationError(
                "Duplicate callId receipt."
            )

        if receipt.dossierId in seen_dossier_ids:
            raise ReceiptVerificationError(
                "Duplicate dossierId receipt."
            )

        seen_receipt_ids.add(
            receipt.receiptId
        )
        seen_call_ids.add(
            receipt.callId
        )
        seen_dossier_ids.add(
            receipt.dossierId
        )

        record = proposals_by_call_id.get(
            receipt.callId
        )

        if record is None:
            raise ReceiptVerificationError(
                f"Unknown callId {receipt.callId!r}."
            )

        proposal = record["proposal"]

        if (
            receipt.dossierId
            != proposal["dossierId"]
        ):
            raise ReceiptVerificationError(
                "Receipt was moved to another dossier."
            )

        if (
            receipt.action
            != proposal["action"]
        ):
            raise ReceiptVerificationError(
                "Receipt action does not match proposal."
            )

        if (
            receipt.proposalDigest
            != record["proposalDigest"]
        ):
            raise ReceiptVerificationError(
                "Receipt proposalDigest does not match "
                "the persisted proposal."
            )

        verify_receipt_signature(
            public_key=public_key,
            evaluation_id=request.evaluationId,
            input_digest=request.inputDigest,
            receipt=receipt,
        )

        verified.append(
            {
                "receipt": receipt,
                "proposal": proposal,
                "proposalDigest": (
                    record["proposalDigest"]
                ),
                "position": record["position"],
            }
        )

    if seen_call_ids != set(
        proposals_by_call_id
    ):
        raise ReceiptVerificationError(
            "One or more proposal receipts are missing."
        )

    verified.sort(
        key=lambda item: item["position"]
    )

    return verified


def load_commit_row(
    evaluation_id: str,
) -> sqlite3.Row | None:
    with database_connection() as connection:
        return connection.execute(
            """
            SELECT *
            FROM commits
            WHERE evaluation_id = ?
            """,
            (evaluation_id,),
        ).fetchone()


def persist_verified_commit(
    request: CommitRequest,
    raw_request: dict[str, Any],
    verified_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Persist receipts and effects exactly once.

    The caller has already verified every receipt, but we still perform
    replay and conflict checks inside one transaction.
    """

    request_fingerprint = sha256_hex(
        raw_request
    )

    request_json = compact_json_text(
        raw_request
    )

    now = utc_now()

    outcomes: list[dict[str, Any]] = []

    for record in verified_records:
        receipt: CommitReceipt = record[
            "receipt"
        ]

        status = (
            "executed"
            if receipt.accepted
            else "rejected"
        )

        outcomes.append(
            {
                "dossierId": receipt.dossierId,
                "callId": receipt.callId,
                "action": receipt.action,
                "proposalDigest": (
                    receipt.proposalDigest
                ),
                "receiptId": receipt.receiptId,
                "status": status,
            }
        )

    response = {
        "profile": PROFILE,
        "evaluationId": request.evaluationId,
        "status": "completed",
        "inputDigest": request.inputDigest,
        "outcomes": outcomes,
    }

    response_json = compact_json_text(
        response
    )

    with _db_lock:
        with database_connection() as connection:
            try:
                connection.execute(
                    "BEGIN IMMEDIATE"
                )

                existing = connection.execute(
                    """
                    SELECT *
                    FROM commits
                    WHERE evaluation_id = ?
                    """,
                    (request.evaluationId,),
                ).fetchone()

                if existing is not None:
                    if (
                        existing["request_fingerprint"]
                        != request_fingerprint
                    ):
                        raise HTTPException(
                            status_code=409,
                            detail=(
                                "evaluationId already has a "
                                "different commit request."
                            ),
                        )

                    if existing["response_json"]:
                        connection.execute(
                            "ROLLBACK"
                        )

                        return json.loads(
                            existing["response_json"]
                        )

                    raise RuntimeError(
                        "Persisted commit is incomplete."
                    )

                connection.execute(
                    """
                    INSERT INTO commits (
                        evaluation_id,
                        request_fingerprint,
                        request_json,
                        response_json,
                        state,
                        created_at,
                        updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        request.evaluationId,
                        request_fingerprint,
                        request_json,
                        response_json,
                        "completed",
                        now,
                        now,
                    ),
                )

                for record, outcome in zip(
                    verified_records,
                    outcomes,
                    strict=True,
                ):
                    receipt: CommitReceipt = record[
                        "receipt"
                    ]

                    receipt_json = (
                        compact_json_text(
                            receipt.model_dump(
                                mode="json"
                            )
                        )
                    )

                    connection.execute(
                        """
                        INSERT INTO receipts (
                            evaluation_id,
                            receipt_id,
                            dossier_id,
                            call_id,
                            action,
                            accepted,
                            proposal_digest,
                            receipt_signature,
                            receipt_json,
                            created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            request.evaluationId,
                            receipt.receiptId,
                            receipt.dossierId,
                            receipt.callId,
                            receipt.action,
                            (
                                1
                                if receipt.accepted
                                else 0
                            ),
                            receipt.proposalDigest,
                            receipt.receiptSignature,
                            receipt_json,
                            now,
                        ),
                    )

                    effect = {
                        "dossierId": receipt.dossierId,
                        "callId": receipt.callId,
                        "action": receipt.action,
                        "proposalDigest": (
                            receipt.proposalDigest
                        ),
                        "receiptId": receipt.receiptId,
                        "accepted": receipt.accepted,
                        "status": outcome["status"],
                    }

                    connection.execute(
                        """
                        INSERT INTO action_effects (
                            evaluation_id,
                            dossier_id,
                            call_id,
                            action,
                            proposal_digest,
                            receipt_id,
                            effect_status,
                            effect_json,
                            created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            request.evaluationId,
                            receipt.dossierId,
                            receipt.callId,
                            receipt.action,
                            receipt.proposalDigest,
                            receipt.receiptId,
                            outcome["status"],
                            compact_json_text(
                                effect
                            ),
                            now,
                        ),
                    )

                connection.execute(
                    """
                    UPDATE evaluations
                    SET
                        state = ?,
                        updated_at = ?
                    WHERE evaluation_id = ?
                    """,
                    (
                        "completed",
                        now,
                        request.evaluationId,
                    ),
                )

                connection.execute(
                    "COMMIT"
                )

            except HTTPException:
                try:
                    connection.execute(
                        "ROLLBACK"
                    )
                except sqlite3.Error:
                    pass

                raise

            except Exception:
                try:
                    connection.execute(
                        "ROLLBACK"
                    )
                except sqlite3.Error:
                    pass

                raise

    return response


def process_commit(
    request: CommitRequest,
    raw_request: dict[str, Any],
) -> dict[str, Any]:
    evaluation = load_evaluation(
        request.evaluationId
    )

    if evaluation is None:
        raise HTTPException(
            status_code=400,
            detail="Unknown evaluationId.",
        )

    if (
        evaluation["input_digest"]
        != request.inputDigest
    ):
        raise HTTPException(
            status_code=409,
            detail=(
                "Commit inputDigest does not match "
                "the persisted evaluation."
            ),
        )

    if not evaluation["response_json"]:
        raise HTTPException(
            status_code=409,
            detail=(
                "Evaluation has no completed proposal response."
            ),
        )

    persisted_propose_response = json.loads(
        evaluation["response_json"]
    )

    if (
        persisted_propose_response.get("status")
        != "awaiting_receipts"
    ):
        existing_commit = load_commit_row(
            request.evaluationId
        )

        if (
            existing_commit is not None
            and existing_commit["response_json"]
        ):
            request_fingerprint = sha256_hex(
                raw_request
            )

            if (
                existing_commit[
                    "request_fingerprint"
                ]
                != request_fingerprint
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "evaluationId already has a "
                        "different commit request."
                    ),
                )

            return json.loads(
                existing_commit[
                    "response_json"
                ]
            )

    existing_commit = load_commit_row(
        request.evaluationId
    )

    if existing_commit is not None:
        request_fingerprint = sha256_hex(
            raw_request
        )

        if (
            existing_commit["request_fingerprint"]
            != request_fingerprint
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "evaluationId already has a different "
                    "commit request."
                ),
            )

        if existing_commit["response_json"]:
            return json.loads(
                existing_commit["response_json"]
            )

    public_key = import_receipt_verifier(
        evaluation["verifier_json"]
    )

    proposal_records = (
        get_persisted_proposals_for_commit(
            request.evaluationId
        )
    )

    verified_records = (
        validate_receipts_against_proposals(
            request=request,
            proposal_records=proposal_records,
            public_key=public_key,
        )
    )

    return persist_verified_commit(
        request=request,
        raw_request=raw_request,
        verified_records=verified_records,
    )


# ============================================================
# Endpoint
# ============================================================

def _provider_hostname() -> str:
    raw = os.environ.get("MAILROOM_AI_URL", "").strip()
    return urlparse(raw).hostname or "unknown"


def _provider_model() -> str:
    return os.environ.get("MAILROOM_AI_MODEL", "").strip()


def _response_format_enabled() -> bool:
    return os.environ.get("MAILROOM_AI_RESPONSE_FORMAT", "1").strip() != "0"


def _make_503_detail(
    error: Exception,
    eval_id: str | None = None,
    dossier_count: int | None = None,
) -> dict[str, Any]:
    """Build a safe 503 detail dict with diagnostic fields."""
    provider_status: int | None = None
    if isinstance(error, AIProviderError):
        provider_status = getattr(error, "status_code", None)

    return {
        "detail": "AI provider failure",
        "error_type": type(error).__name__,
        "provider_status": provider_status,
        "reason": "The external decision provider could not complete.",
    }


@router.post("/mailroom-agent")
async def mailroom_agent(
    body: dict[str, Any],
) -> dict[str, Any]:
    """
    Layer 1 + 2 + 3:
    - validate envelopes
    - canonical hashing
    - durable persistence
    - replay and conflict detection
    - AI proposal generation with caching
    - frozen action schema validation
    - Ed25519 receipt verification
    - atomic commit with all-or-nothing semantics
    """

    operation = body.get("operation")

    if operation == "propose":
        try:
            request = ProposeRequest.model_validate(
                body
            )
        except ValidationError as error:
            raise HTTPException(
                status_code=422,
                detail=safe_validation_errors(
                    error
                ),
            ) from error

        eval_id = request.evaluationId
        dossier_count = len(request.dossiers)
        start_time = time_module.monotonic()

        try:
            result = save_new_evaluation(
                request=request,
                raw_request=body,
            )

            if result["kind"] == "exact_replay":
                persisted_response = (
                    load_persisted_propose_response(
                        request.evaluationId
                    )
                )

                if persisted_response is not None:
                    return persisted_response

                return (
                    build_and_persist_propose_response(
                        request
                    )
                )

            return (
                build_and_persist_propose_response(
                    request
                )
            )

        except HTTPException:
            raise

        except ProposalValidationError as error:
            hostname = _provider_hostname()
            model = _provider_model()
            elapsed = time_module.monotonic() - start_time
            logger.error(
                "error=ProposalValidationError evalId=%s "
                "hostname=%s model=%s elapsed=%.1fs "
                "dossier_count=%d response_format=%s",
                eval_id, hostname, model, elapsed,
                dossier_count, _response_format_enabled(),
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "message": (
                        "The AI produced a proposal "
                        "that violated the action contract."
                    ),
                    "error": str(error),
                },
            ) from error

        except AIOutputError as error:
            hostname = _provider_hostname()
            model = _provider_model()
            elapsed = time_module.monotonic() - start_time
            logger.error(
                "error=AIOutputError evalId=%s "
                "hostname=%s model=%s elapsed=%.1fs "
                "dossier_count=%d response_format=%s",
                eval_id, hostname, model, elapsed,
                dossier_count, _response_format_enabled(),
            )
            raise HTTPException(
                status_code=502,
                detail={
                    "message": (
                        "The AI did not produce usable "
                        "structured proposal output."
                    ),
                    "error": str(error),
                },
            ) from error

        except AIProviderError as error:
            hostname = getattr(error, "provider_hostname", _provider_hostname())
            model = getattr(error, "model", _provider_model())
            provider_status = getattr(error, "status_code", None)
            preview = getattr(error, "preview", "")
            attempt = getattr(error, "attempt", 0)
            response_format = getattr(error, "response_format_enabled", _response_format_enabled())
            elapsed = time_module.monotonic() - start_time
            logger.error(
                "error=AIProviderError evalId=%s "
                "hostname=%s model=%s status=%s elapsed=%.1fs "
                "dossier_count=%d response_format=%s attempt=%d "
                "preview=%.500s",
                eval_id, hostname, model, provider_status,
                elapsed, dossier_count, response_format,
                attempt, preview,
            )
            raise HTTPException(
                status_code=503,
                detail=_make_503_detail(
                    error, eval_id, dossier_count,
                ),
            ) from error

        except (
            RuntimeError,
            json.JSONDecodeError,
            httpx.HTTPError,
            sqlite3.Error,
        ) as error:
            hostname = _provider_hostname()
            model = _provider_model()
            elapsed = time_module.monotonic() - start_time
            logger.exception(
                "error=%s evalId=%s hostname=%s model=%s "
                "elapsed=%.1fs dossier_count=%d",
                type(error).__name__, eval_id,
                hostname, model, elapsed, dossier_count,
            )
            raise HTTPException(
                status_code=503,
                detail=_make_503_detail(
                    error, eval_id, dossier_count,
                ),
            ) from error

    if operation == "commit":
        try:
            request = CommitRequest.model_validate(
                body
            )
        except ValidationError as error:
            raise HTTPException(
                status_code=422,
                detail=safe_validation_errors(
                    error
                ),
            ) from error

        try:
            return process_commit(
                request=request,
                raw_request=body,
            )

        except HTTPException:
            raise

        except ReceiptVerificationError as error:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": (
                        "Receipt verification failed."
                    ),
                    "error": str(error),
                },
            ) from error

        except (
            RuntimeError,
            json.JSONDecodeError,
            sqlite3.Error,
        ) as error:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": (
                        "Commit processing could not "
                        "complete safely."
                    ),
                    "error": str(error),
                },
            ) from error

    raise HTTPException(
        status_code=400,
        detail=(
            "operation must be either "
            "'propose' or 'commit'."
        ),
    )
