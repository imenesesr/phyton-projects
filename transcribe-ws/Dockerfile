FROM python:3.9-slim

# 1. Instalar solo dependencias críticas del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Instalar torch CPU (versión ligera) antes que otros paquetes
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# 3. Instalar el resto de dependencias
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p /app/temp

EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app", "--workers", "2", "--timeout", "300"]