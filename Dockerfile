FROM python:3.11-slim

WORKDIR /app

# Добавили libspatialindex-dev для работы карт (osmnx)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]