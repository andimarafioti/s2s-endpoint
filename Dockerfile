FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    INTERNAL_WS_HOST=127.0.0.1 \
    INTERNAL_WS_PORT=9000 \
    S2S_REPO_DIR=/opt/speech-to-speech

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsndfile1 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install uv

# Clone speech-to-speech and install its dependencies the way the repo expects
RUN git clone --depth 1 https://github.com/huggingface/speech-to-speech.git ${S2S_REPO_DIR} && \
    cd ${S2S_REPO_DIR} && \
    git fetch --depth 1 origin ${S2S_REF} && \
    git checkout ${S2S_REF} && \
    uv sync --no-dev

COPY app /app/app

EXPOSE 7860

CMD ["uv", "run", "--directory", "/app", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
