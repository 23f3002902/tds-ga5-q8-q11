import re

from fastapi import FastAPI, Request

from q3_guardrail import router as q3_router
from q4_skill_scanner import router as q4_router
from q8_redteam_guardrail import create_required_files, router as q8_router
from q9_mailroom import router as q9_router
from q10_a2a_invoice_agent import (
    install_q10_exception_handlers,
    router as q10_router,
)
from q11_incident_agent import router as q11_router


app = FastAPI(title="TDS GA5 Services", version="1.1.0")


@app.middleware("http")
async def normalize_a2a_path(request: Request, call_next):
    """Avoid redirects or 404s when an A2A client joins base paths naively."""
    path = request.scope.get("path") or "/"
    if path.startswith("/a2a/") or path.startswith("//"):
        normalized = re.sub(r"/{2,}", "/", path)
        if len(normalized) > 1 and normalized.endswith("/"):
            normalized = normalized.rstrip("/")
        if normalized != path:
            request.scope["path"] = normalized
            request.scope["raw_path"] = normalized.encode("utf-8")
    return await call_next(request)


@app.on_event("startup")
def seed_q8_files() -> None:
    try:
        create_required_files()
    except OSError:
        pass


app.include_router(q3_router)
app.include_router(q4_router)
app.include_router(q8_router)
app.include_router(q9_router)
app.include_router(q10_router)
install_q10_exception_handlers(app)
app.include_router(q11_router)


@app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "questions": [3, 4, 8, 9, 10, 11],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
