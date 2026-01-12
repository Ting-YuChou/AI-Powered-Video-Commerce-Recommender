# AI-Powered Video Commerce Recommender - Production Dockerfile
# ==============================================================
# Multi-stage Docker build for efficient, secure, production deployment

# Build arguments
ARG PYTHON_VERSION=3.10
ARG BUILD_ENV=production
ARG DEBIAN_FRONTEND=noninteractive

# =============================================================================
# Stage 1: Base Dependencies
# =============================================================================
FROM python:${PYTHON_VERSION}-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    pkg-config \
    # Media processing
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    # OCR dependencies
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    # Image processing
    libopencv-dev \
    libglib2.0-0 \
    libgtk-3-dev \
    # Audio processing
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    # Network utilities
    curl \
    wget \
    # Process monitoring
    procps \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/*

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser -d /app -s /bin/bash appuser

# Set working directory
WORKDIR /app

# =============================================================================
# Stage 2: Python Dependencies
# =============================================================================
FROM base as python-deps

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel \
    cython

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# =============================================================================
# Stage 3: Application Build
# =============================================================================
FROM base as app-build

# Copy Python dependencies from previous stage
COPY --from=python-deps /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Create necessary directories
RUN mkdir -p /app/models /app/data /app/logs /app/uploads /app/tmp

# Copy application code
COPY . /app/

# Install the application (if setup.py exists)
RUN if [ -f setup.py ]; then pip install --no-cache-dir -e .; fi

# Download and cache models (optional)
ARG DOWNLOAD_MODELS=false
RUN if [ "$DOWNLOAD_MODELS" = "true" ]; then \
    python -c "from transformers import CLIPModel, CLIPProcessor; \
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14', cache_dir='/app/models'); \
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14', cache_dir='/app/models')"; \
    fi

# =============================================================================
# Stage 4: Production Runtime
# =============================================================================
FROM base as production

# Copy Python dependencies
COPY --from=app-build /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application
COPY --from=app-build /app /app

# Change ownership to non-root user
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MODEL_CACHE_DIR=/app/models
ENV DATA_UPLOAD_DIR=/app/uploads

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8000}/health || exit 1

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Stage 5: Development Build
# =============================================================================
FROM app-build as development

# Install development dependencies
COPY requirements-dev.txt* ./
RUN if [ -f requirements-dev.txt ]; then pip install --no-cache-dir -r requirements-dev.txt; fi

# Install additional development tools
RUN pip install --no-cache-dir \
    jupyter \
    jupyterlab \
    ipython \
    black \
    isort \
    flake8 \
    pytest \
    pytest-asyncio \
    pytest-cov

# Enable Jupyter extensions
RUN jupyter lab build

# Development environment variables
ENV ENVIRONMENT=development
ENV API_DEBUG=true
ENV API_RELOAD=true
ENV MONITORING_LOG_LEVEL=DEBUG

# Don't switch to non-root user in development for flexibility
# Keep as root to allow package installation and debugging

# Expose additional ports for development
EXPOSE 8000 8888

# Development command (can be overridden)
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# =============================================================================
# Stage 6: GPU-Enabled Runtime
# =============================================================================
FROM nvidia/cuda:11.8-runtime-ubuntu20.04 as gpu-runtime

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    tesseract-ocr \
    tesseract-ocr-eng \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy application and dependencies
COPY --from=app-build /root/.local /root/.local
COPY --from=app-build /app /app

# Set environment variables
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0
ENV MODEL_DEVICE=cuda

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install FAISS GPU
RUN pip install --no-cache-dir faiss-gpu

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
RUN chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# =============================================================================
# Build Instructions and Examples
# =============================================================================

# Build production image:
# docker build --target production -t video-commerce:latest .

# Build development image:
# docker build --target development -t video-commerce:dev .

# Build GPU-enabled image:
# docker build --target gpu-runtime -t video-commerce:gpu .

# Build with model downloading:
# docker build --build-arg DOWNLOAD_MODELS=true -t video-commerce:with-models .

# Build for specific Python version:
# docker build --build-arg PYTHON_VERSION=3.10 -t video-commerce:py310 .

# Multi-platform build:
# docker buildx build --platform linux/amd64,linux/arm64 -t video-commerce:latest .

# =============================================================================
# Docker Compose Integration
# =============================================================================

# The following docker-compose.yml snippet shows how to use this Dockerfile:
#
# services:
#   app:
#     build:
#       context: .
#       dockerfile: Dockerfile
#       target: production
#       args:
#         - PYTHON_VERSION=3.9
#         - BUILD_ENV=production
#         - DOWNLOAD_MODELS=false
#     environment:
#       - ENVIRONMENT=production
#     volumes:
#       - ./models:/app/models
#       - ./uploads:/app/uploads
#     ports:
#       - "8000:8000"

# =============================================================================
# Security Best Practices
# =============================================================================

# 1. Non-root user: Application runs as 'appuser', not root
# 2. Minimal base image: Uses slim Python image to reduce attack surface
# 3. Security scanning: Run security scans on the built image
#    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
#      -v $(pwd):/tmp/.cache/ aquasec/trivy image video-commerce:latest
# 4. Secrets management: Don't embed secrets in the image
# 5. Regular updates: Rebuild images regularly for security patches

# =============================================================================
# Optimization Tips
# =============================================================================

# 1. Layer caching: Dependencies are installed before copying source code
# 2. Multi-stage build: Reduces final image size by excluding build dependencies
# 3. .dockerignore: Use .dockerignore to exclude unnecessary files
# 4. Model caching: Pre-download models during build if needed
# 5. Cleanup: Remove package manager caches and temporary files

# =============================================================================
# Monitoring and Debugging
# =============================================================================

# Access running container:
# docker exec -it container_name bash

# Check container resources:
# docker stats container_name

# View container logs:
# docker logs -f container_name

# Inspect image layers:
# docker history video-commerce:latest

# Check image size:
# docker images video-commerce

# Security scan:
# docker scout cves video-commerce:latest

