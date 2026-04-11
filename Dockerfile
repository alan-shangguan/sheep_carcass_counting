# ---- Unified minimal image: CPU-only, no GPU, minimal deps --------
FROM python:3.10-slim

WORKDIR /app

# Minimal system libs for OpenCV + pip build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libglib2.0-0 \
    libxcb1 \
    libx11-6 \
    libxext6 \
    libsm6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (CPU-only, no GPU support)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ app/

# Videos and model weights are mounted at runtime (see docker-compose.yml)
# so we just create the expected mount-point directories.
RUN mkdir -p /app/videos /app/weights

# Expose the FastAPI port
EXPOSE 8000

# Environment variable defaults – overridable via docker-compose or -e flags.
# Paths are relative to WORKDIR (/app).
ENV VIDEO_PATH="videos/Event20260123020157006.mp4" \
    MODEL_PATH="weights/best_openvino_model" \
    GATE_Y_RATIO="0.5" \
    DEAD_BAND_RATIO="0.05" \
    MIN_HITS="3" \
    COUNT_DIRECTION="any" \
    MODEL_CLASSES="" \
    JPEG_QUALITY="80"

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
