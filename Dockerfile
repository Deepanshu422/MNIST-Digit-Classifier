# Step 1: Base Python image (slim keeps it lightweight)
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory inside the container
WORKDIR /app

# Step 2: Install system dependencies required for image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 3: Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy application source code and artifacts
COPY src/ ./src
COPY artifacts/ ./artifacts

# Step 5: Expose ports
# 8000 for FastAPI Backend, 7860 for Gradio Frontend
EXPOSE 8000 7860

# Step 6: Default command (runs the interactive frontend by default)
CMD ["python", "src/frontend/app.py"]