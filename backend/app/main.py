import asyncio

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import Base, engine
from app.routers import search
from app.routers import timeline
from app.routers import graph
from app.routers import events
from app.routers import intervention
from app.routers import verification
from app.routers import ws_events
from app.routers import factcheck
from app.routers import radar

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search.router)
app.include_router(timeline.router)
app.include_router(graph.router)
app.include_router(events.router)
app.include_router(intervention.router)
app.include_router(verification.router)
app.include_router(ws_events.router)
app.include_router(factcheck.router)
app.include_router(radar.router)

@app.on_event("startup")
async def _start_background_jobs():
    # Fire-and-forget: this task lives for the process lifetime, not tied to
    # any single request. See radar.run_intervention_scheduler_forever for
    # why this can't just piggyback on a request like _maybe_ingest does.
    asyncio.create_task(radar.run_intervention_scheduler_forever())

@app.get("/")
def root():
    return {"status": "ok"}