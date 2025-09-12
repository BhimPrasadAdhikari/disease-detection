# Use official TensorFlow image
FROM tensorflow/tensorflow:2.15.0

# Set working directory
WORKDIR /app

# Copy only necessary files
COPY app.py /app
COPY requirements.txt /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8080

# Start Flask with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
