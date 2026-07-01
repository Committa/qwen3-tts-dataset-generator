FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VIRTUALENVS_IN_PROJECT=false

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip curl ca-certificates git \
        sox ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3.12 - --version 1.8.4 \
    && ln -s /root/.local/bin/poetry /usr/local/bin/poetry

WORKDIR /workspace

COPY pyproject.toml ./
COPY poetry.lock* ./
RUN poetry install --no-root --without dev

COPY src/ ./src/
COPY config.yaml ./
COPY inputs/ ./inputs/

RUN mkdir -p workspace/raw_wav workspace/accepted_wav workspace/rejected output logs inputs

ENTRYPOINT ["poetry", "run", "gen-dataset"]
CMD []