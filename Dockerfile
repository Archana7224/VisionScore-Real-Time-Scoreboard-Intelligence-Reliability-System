# VisionScore dashboard - persistent container image.
# Runs the REAL OpenCV + EasyOCR pipeline behind a Flask/gunicorn server.
# Suitable for any container host (Render, Railway, Fly.io, Hugging Face Spaces).

FROM python:3.12-slim

# System libraries required by OpenCV (headless) and torch/easyocr runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install Python dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download the EasyOCR English models into the image so the first
# request is fast and models survive on ephemeral container filesystems.
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False, verbose=False)"

# Copy the application source.
COPY . .

# Container hosts inject $PORT; gunicorn binds to it.
# ONE worker + threads keeps the in-memory job registry shared across
# requests (the dashboard polls /api/status), and the long timeout allows
# a full video to be processed within a single request-handling window.
EXPOSE 8080
CMD gunicorn app:app \
    --bind 0.0.0.0:${PORT} \
    --workers 1 \
    --threads 8 \
    --timeout 1200
