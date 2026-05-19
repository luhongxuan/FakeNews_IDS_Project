FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ git curl \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install --upgrade pip

# ── 安裝套件 ──────────────────────────────
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY graph_analysis/requirements.txt ./requirements-graph.txt
RUN pip install --no-cache-dir -r requirements-graph.txt

# ── 複製程式碼 ────────────────────────────
COPY backend/ .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]