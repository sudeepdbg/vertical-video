FROM python:3.11-slim

# Install FFmpeg and OpenCV system dependencies
# FIX: libgl1-mesa-glx replaced with libgl1 for Debian Bookworm+
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt boto3

# Copy your code
COPY verticalize.py worker.py .

# Command to run when the container starts
CMD ["python", "worker.py"]
