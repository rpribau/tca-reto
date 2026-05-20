# =========================================================
# Dockerfile for Assist v10 API & Dashboard
# =========================================================

FROM python:3.10-slim

# Evitar que Python escriba archivos .pyc y forzar salida sin buffer para logs limpios
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar dependencias del sistema necesarias para LightGBM y compilaciones básicas
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar configuración del proyecto y metadatos
COPY pyproject.toml .
COPY conf/ ./conf/

# Copiar el código fuente
COPY src/ ./src/

# Crear directorios para datos y tracking para asegurar permisos correctos
RUN mkdir -p data/01_raw data/02_intermediate data/03_primary data/06_models data/09_tracking mlruns

# Exponer el puerto de la API FastAPI
EXPOSE 8000

# Ejecutar la API
CMD ["uvicorn", "assist_v10.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
