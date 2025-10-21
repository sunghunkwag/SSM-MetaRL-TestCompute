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

# Install the wheel
RUN pip install --no-cache-dir /tmp/*.whl

# Copy the main script to run
COPY main.py /app/main.py

# Set the entrypoint
ENTRYPOINT ["python", "main.py"]
