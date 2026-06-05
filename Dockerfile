FROM python:3.10-slim
RUN apt-get update && apt-get install -y ffmpeg libgl1-mesa-glx libglib2.0-0
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt boto3
COPY verticalize.py worker.py .
CMD ["python", "worker.py"]
