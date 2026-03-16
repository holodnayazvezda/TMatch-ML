# syntax=docker/dockerfile:1
ARG PY_IMAGE=python:3.12-slim

FROM ${PY_IMAGE} AS api

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app:/app/src

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

COPY pyproject.toml ./
COPY uv.lock ./
COPY src ./src

RUN uv sync --no-dev

EXPOSE 8080

CMD ["uv", "run", "api"]
