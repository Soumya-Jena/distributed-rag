# syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

WORKDIR /build
RUN python -m venv /opt/venv
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu \
       torch==2.13.0
COPY requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt

FROM python:3.11-slim AS runtime

RUN groupadd --gid 10001 rag \
    && useradd --uid 10001 --gid rag --no-log-init --create-home rag

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV HF_HOME=/home/rag/.cache/huggingface

COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY --chown=rag:rag src ./src
RUN mkdir -p /home/rag/.cache/huggingface \
    && chown -R rag:rag /home/rag /app

USER rag
STOPSIGNAL SIGTERM
