FROM python:3.10-slim

WORKDIR /app

# Combine system deps and clean up in one layer to save space
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libsndfile1 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Create storage with correct permissions
RUN mkdir -p storage/documents storage/temp storage/qdrant_storage

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

EXPOSE 8000

# Using the list form for the CMD is preferred
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]