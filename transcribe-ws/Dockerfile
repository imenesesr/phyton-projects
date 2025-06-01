FROM python:3.11-slim

# 1. Instalar solo dependencias críticas del sistema
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    espeak-ng \
    libespeak-ng1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Copiar requirements.txt primero para aprovechar el caché de Docker
COPY requirements.txt .

# 3. Instalar dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch==2.1.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu 

# 4. Copiar el resto del código
COPY . .

# Crear directorios temporales
RUN mkdir -p /app/temp

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app", "--workers", "2", "--timeout", "300"]