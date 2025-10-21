# ---- Build Stage ----
# Use a full Python image to build the wheel
FROM python:3.9 as builder

WORKDIR /app

# Copy only necessary files for building
COPY pyproject.toml README.md /app/
COPY core /app/core
COPY meta_rl /app/meta_rl
COPY env_runner /app/env_runner
COPY adaptation /app/adaptation

# Install build dependencies and build the wheel
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build && \
    python -m build --wheel --outdir dist .

# ---- Final Stage ----
# Use a slim image for the final runtime
FROM python:3.9-slim

WORKDIR /app

# Copy the built wheel from the builder stage
COPY --from=builder /app/dist/*.whl /tmp/

# Install the wheel and runtime dependencies (like gymnasium if needed by scripts)
# Note: Ensure all runtime deps are covered by the wheel or install them here
RUN pip install --no-cache-dir /tmp/*.whl && \
    rm /tmp/*.whl # Clean up the wheel file

# Copy application scripts (main, experiments, etc.)
COPY main.py /app/main.py
COPY experiments /app/experiments

# No ENTRYPOINT, allow user to specify script via docker run command
# Example: docker run <image_name> python main.py --env_name Pendulum-v1
# Example: docker run <image_name> python experiments/quick_benchmark.py

# Default command (optional, can be overridden)
CMD ["python", "main.py", "--help"]
