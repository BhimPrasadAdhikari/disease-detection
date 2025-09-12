# Use official TensorFlow image
FROM tensorflow/tensorflow:2.15.0

# Set working directory
WORKDIR /app

# Copy only requirements first (so dependencies are cached)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (except those in .dockerignore)
COPY . .

# Expose port
EXPOSE 8080

# Start Flask with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
