# Build stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files first (better caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (non-editable for production)
RUN uv sync --frozen --no-dev --no-editable

# Copy application files - static files first for better caching
COPY README.md chainlit.md ./
COPY public ./public
COPY src ./src

# Runtime stage
FROM python:3.12-slim-bookworm

WORKDIR /app

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/chainlit.md /app/README.md ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/src ./src

# Environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CHAINLIT_HOST=0.0.0.0 \
    CHAINLIT_PORT=8000

EXPOSE 8000

CMD ["chainlit", "run", "src/pairreader/__main__.py", "--host", "0.0.0.0", "--port", "8000"]