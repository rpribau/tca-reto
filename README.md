# TCA-Reto: Assist v10 (Gestión Hospitalaria con IA)

Sistema predictivo de IA para optimizar la gestión hospitalaria mediante pronóstico de tiempos de espera y detección de ausentismo (No-Shows) en citas, escalado con Kedro y servido en FastAPI para uso en Azure.

## 🛠 Stack Tecnológico
- **Orquestación de Datos**: [Kedro](https://kedro.readthedocs.io/)
- **Modelado**: XGBoost (Time Series) y CatBoost (Clasificación Categórica)
- **API**: [FastAPI](https://fastapi.tiangolo.com/) y Pydantic (Type Hints)
- **Almacenamiento**: DataFrames en formato `.parquet`
- **Despliegue**: Docker y Microsoft Azure

---

## 👥 División del Trabajo 

### 1. Data Engineer (Arquitectura Kedro y Procesamiento)
**Responsabilidades**: 
- Diseñar y desarrollar los nodos iniciales de extracción, limpieza y cruce de datos (`HOSPAC`, `HOSMPI`, `NOTAMEDICAURG`, `TRIAGE`, `HOSAGD`).
- Asegurar que los datos crudos `.parquet` ingresen correctamente al catalog y salgan procesados de la capa `data_engineering`.
- **Ubicación clave**: `src/assist_v10/pipelines/data_engineering/`

### 2. Data Scientist A - HIS-05 (Monitor de Saturación)
**Responsabilidades**: 
- Entrenar y afinar el modelo de **Time Series Forecasting** para predecir los tiempos de espera.
- Implementar validación cruzada para series de tiempo (`TimeSeriesSplit`) y extraer features temporales (estacionalidad, día de la semana).
- Entrenar el modelo base sugerido (`XGBRegressor`) y guardar las métricas de rendimiento en el catálogo (`evaluation_metrics_his05`).
- **Ubicación clave**: `src/assist_v10/pipelines/data_science/his05/`

### 3. Data Scientist B - HIS-10 (No-Show Guard)
**Responsabilidades**: 
- Desarrollar el clasificador de **Propensión de Inasistencia** para optimización de agendas.
- Analizar el balanceo de clases (manejar el 15-25% de ausentismo) y procesar características categóricas altas (códigos postales, doctores, especialidades).
- Entrenar el modelo base sugerido (`CatBoostClassifier`) el cual es idóneo para variables no numéricas, guardando métricas (`evaluation_metrics_his10`).
- **Ubicación clave**: `src/assist_v10/pipelines/data_science/his10/`

### 4. Backend/MLOps Engineer (API y Pydantic)
**Responsabilidades**: 
- Mantener y mejorar los servicios web construidos en [FastAPI](src/assist_v10/api/main.py).
- Manejar la interacción entre los modelos estáticos generados por Kedro (`PickleDataSet`) e inyectarlos localmente usando `KedroSession.create()`.
- Fortalecer los [Type Hints y Validaciones Pydantic](src/assist_v10/api/schemas.py) para que todas las peticiones desde el frontend reaccionen limpiamente con HTTP Exceptions en caso de fallo.
- **Ubicación clave**: `src/assist_v10/api/`

### 5. DevOps & Cloud Engineer (Azure Deployment)
**Responsabilidades**: 
- Construir el `Dockerfile` que empaquete las dependencias (Kedro + FastAPI + Modelos entrenados).
- Diseñar la estrategia de entrega continua (CI/CD) para Azure basándose en el resguardo seguro del código (sin subir la carpeta `/data`).
- Vigilar y monitorear los contenedores desplegados usando el endpoint preventivo `/v1/health`.
- **Tecnologías clave**: Docker, Azure Web App / AKS, GitHub Actions / Azure DevOps.