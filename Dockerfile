# base image
FROM python:3.9-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# builder
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# copying the essentials
COPY pyproject.toml /app/
COPY src /app/src
COPY config app/config

# build wheels, install packages
RUN pip install --update pip \
    && pip wheels --no-cache-dir

# test image
FROM base AS test

COPY data /app/data
COPY tests /app/tests

CMD ["pytest", "-q"]

# train
FROM base AS train

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

COPY data /app/data
COPY src /app/src

CMD ["python","src/cli.py","process"]
CMD ["python","src/cli.py","train"]

# production
FROM base AS production

RUN addgroup --system app && adduser --system --ingroup app app

COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* \
    && rm -rf /wheels

# 🔴 COPY RUNTIME FILES
COPY config /app/config
COPY logs /app/logs

# After copying source (This ensures USER app can write there.)
RUN mkdir -p /app/logs && chown -R app:app /app

USER app

EXPOSE 8000
CMD ["uvicorn","src.cli:app", "--host", "0.0.0.0", "--port", "8000"]
