# Use a lightweight Python base image to reduce container size
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install essential system dependencies and clean up apt cache to save space
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY training/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY training/ /app/training/

# Set PYTHONPATH so Python can locate modules in the /app directory
ENV PYTHONPATH="/app:/app/data_pipeline"

# Default command to run the pipeline
CMD ["python", "training/train.py"]
