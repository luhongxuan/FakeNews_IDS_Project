"""Local (non-Docker) dev entry point.

The checked-in backend/.env points DATABASE_URL at the docker-compose
service name "db", which only resolves inside the compose network. For
running the backend directly on the host against a Postgres reachable at
localhost (e.g. `docker compose up -d db`), this sets a localhost default
before python-dotenv loads .env -- python-dotenv does not override
variables that are already set in the process environment.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "postgresql://admin:secret@localhost:15432/misinfo_db")

# backend/.env sets PHEME_PATH="../data/raw/pheme", relative to the Docker
# WORKDIR (/app == backend/). Outside Docker this is only correct if cwd
# happens to be backend/; pin an absolute value so it resolves the same way
# regardless of how this script is launched.
_ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("PHEME_PATH", str(_ROOT / "data" / "raw" / "pheme"))

# The routers import `graph_analysis` assuming the Docker layout where the
# repo root is mounted at /app (see routers/graph.py, routers/events.py
# `sys.path.append("/app")`). Add the real repo root instead for local runs.
sys.path.insert(0, str(_ROOT))

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
