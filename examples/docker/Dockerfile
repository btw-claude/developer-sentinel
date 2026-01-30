# Developer Sentinel Docker Image
# Multi-stage build for minimal image size

# =============================================================================
# Stage 1: Build dependencies
# =============================================================================
FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only dependency files first (better layer caching)
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --upgrade pip && \
    pip install --no-cache-dir .

# =============================================================================
# Stage 2: Runtime image
# =============================================================================
FROM python:3.11-slim as runtime

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --shell /bin/bash sentinel

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create directories for runtime data
RUN mkdir -p /app/orchestrations /app/workdir /app/logs && \
    chown -R sentinel:sentinel /app

# Copy application source
COPY --chown=sentinel:sentinel src/ ./src/

# Switch to non-root user
USER sentinel

# Default environment variables
ENV SENTINEL_ORCHESTRATIONS_DIR=/app/orchestrations \
    SENTINEL_AGENT_WORKDIR=/app/workdir \
    SENTINEL_AGENT_LOGS_DIR=/app/logs \
    SENTINEL_LOG_LEVEL=INFO \
    SENTINEL_POLL_INTERVAL=60

# Health check - verify sentinel module imports and liveness check passes
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from sentinel.health import HealthChecker; checker = HealthChecker(); result = checker.check_liveness(); exit(0 if result.status == 'healthy' else 1)" || exit 1

# Run the sentinel
ENTRYPOINT ["sentinel"]
CMD ["--log-level", "INFO"]
