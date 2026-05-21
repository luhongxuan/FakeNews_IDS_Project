from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import Base, engine
from app.routers import search
from app.routers import timeline
from app.routers import graph
from app.routers import events

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

@app.get("/")
def root():
    return {"status": "ok"}