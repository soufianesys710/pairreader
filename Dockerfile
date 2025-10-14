# Build stage: it is important to use same python version as in pyproject.toml, as we deactivate python download
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

# UV_PYTHON_DOWNLOADS to use system python, not download a new one
# UV_COMPILE_BYTECODE to compile bytecode, faster startup
ENV UV_PYTHON_DOWNLOADS=0 UV_COMPILE_BYTECODE=1

WORKDIR /app

# Install dependencies first (no project installation)
COPY uv.lock ./
COPY pyproject.toml ./
COPY README.md ./
RUN uv sync --locked --no-install-project --no-dev

# Install project now
COPY chainlit.md ./
COPY public ./public
COPY src ./src
RUN uv sync --locked --no-dev

# Runtime stage: it is important to use same python version as in pyproject.toml and the builder
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy virtual environment and application
COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/public ./public
COPY --from=builder /app/chainlit.md ./

# Environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app/src" \
    CHAINLIT_HOST=0.0.0.0 \
    CHAINLIT_PORT=8000

EXPOSE 8000

CMD ["chainlit", "run", "src/pairreader/__main__.py", "--host", "0.0.0.0", "--port", "8000"]