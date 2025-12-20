FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements-hf.txt .
RUN pip install --no-cache-dir -r requirements-hf.txt

# Copy application files
COPY web_app.py .
COPY templates/ templates/
COPY models/ models/

# HF Spaces requires port 7860
ENV PORT=7860
EXPOSE 7860

# Run the Flask app
CMD ["python", "web_app.py"]
