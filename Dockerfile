FROM tensorflow/tensorflow:2.15.0

WORKDIR /app

COPY requirements.txt .

# Upgrade pip first
RUN pip install --upgrade pip setuptools

# Install dependencies without uninstall conflicts
RUN pip install --no-cache-dir --ignore-installed -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
