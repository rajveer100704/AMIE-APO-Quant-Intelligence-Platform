# Base Image
FROM python:3.11-slim

# System Dependencies (including C++ build tools for optimized solver)
RUN apt-get update && apt-get install -y \
    build-essential \
    libopenblas-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Working Directory
WORKDIR /app

# Copy Requirements and Install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Source Code
COPY . .

# Install the Optimized C++ Solver
RUN pip install .

# Environment Variables
ENV EXECUTION_MODE=DRY
ENV REDIS_HOST=localhost
ENV REDIS_PORT=6379

# Expose API and Dask Dashboard Ports
EXPOSE 8000
EXPOSE 8787

# Launch Application
ENTRYPOINT ["uvicorn", "src.api.server.py:app", "--host", "0.0.0.0", "--port", "8000"]
