FROM python:3.10-slim

# Install system dependencies for OpenCV and PyMuPDF
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the server files
COPY . .

# Hugging Face Spaces runs on port 7860 by default
EXPOSE 7860

# Run Flask using gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app", "--timeout", "120"]
