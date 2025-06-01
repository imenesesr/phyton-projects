FROM python:3.9-bullseye  # Debian con más paquetes preinstalados

RUN apt-get update && \
    apt-get install -y ffmpeg libgomp1 espeak-ng && \
    apt-get clean

WORKDIR /app

# 2. Copiar requirements.txt primero para aprovechar el caché de Docker
COPY requirements.txt .

# 3. Actualizar pip e instalar dependencias de Python
#RUN pip install --upgrade pip && \
#    pip install --no-cache-dir torch==2.1.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
#    BLIS_ARCH="generic" pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copiar el resto del código
COPY . .

# Crear directorios temporales
RUN mkdir -p /app/temp

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app", "--workers", "2", "--timeout", "300"]