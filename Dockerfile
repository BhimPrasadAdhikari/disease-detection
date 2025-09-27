FROM python:3.11-slim
FROM tensorflow/tensorflow:2.15.0
# Set working directory
WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    libgl1 libglib2.0-0 wget curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install deps
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt


# Copy project files
COPY . .

# Expose Cloud Run port
EXPOSE 8080

# Gunicorn entrypoint (env expansion works via sh -c)
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-8080} app:app"]
