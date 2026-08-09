from fastapi import FastAPI

from q8_redteam_guardrail import create_required_files, router as q8_router
from q9_mailroom import router as q9_router
from q10_a2a_invoice_agent import (
    install_q10_exception_handlers,
    router as q10_router,
)
from q11_incident_agent import router as q11_router


app = FastAPI(title="TDS GA5 Q8-Q11", version="1.0.0")


@app.on_event("startup")
def seed_q8_files() -> None:
    try:
        create_required_files()
    except OSError:
        pass


app.include_router(q8_router)
app.include_router(q9_router)
app.include_router(q10_router)
install_q10_exception_handlers(app)
app.include_router(q11_router)


@app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "questions": [8, 9, 10, 11],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
