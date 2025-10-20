FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir build && \
    python -m build && \
    pip install --no-cache-dir dist/*.whl

ENTRYPOINT ["python", "main.py"]
