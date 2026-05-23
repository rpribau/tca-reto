# Streamlit - TCA Software Solutions

## Instalación

Asegúrate de instalar las dependencias:

```bash
pip install -r requirements.txt
```

## Ejecución

Ejecuta la aplicación Streamlit desde el directorio raíz del proyecto:

```bash
streamlit run src/assist_v10/streamlit_app.py
```

La aplicación se abrirá automáticamente en tu navegador (por defecto: http://localhost:8501)

## Credenciales de Prueba

Para acceder a la aplicación, utiliza las siguientes credenciales:

- **Usuario:** admin | **Contraseña:** admin123
- **Usuario:** demo | **Contraseña:** demo123

## Características

### Página de Inicio
- Información sobre TCA Software Solutions
- Descripción de las soluciones disponibles
- Métricas clave de performance

### HIS-10: No-Show Guard
- Predicción de inasistencias a citas
- Formulario para ingreso de datos del paciente
- Recomendaciones automáticas basadas en riesgo
- Historial de predicciones guardado en SQLite

### HIS-05: Monitor de Tiempos de Espera
- Estimación de tiempos de espera en áreas hospitalarias
- Análisis por nivel de Triage
- Alertas de saturación
- Historial completo de predicciones

## Estructura de Archivos

```
src/assist_v10/
├── streamlit_app.py      # Aplicación principal de Streamlit
├── auth.py               # Sistema de autenticación
├── db.py                 # Base de datos SQLite
└── predictions.db        # Base de datos de predicciones (se crea automáticamente)
```

## Base de Datos

La aplicación crea automáticamente una base de datos SQLite con las siguientes tablas:

- `users`: Registro de usuarios
- `predictions_his10`: Historial de predicciones HIS-10
- `predictions_his05`: Historial de predicciones HIS-05

## Tema de Colores Corporativos

- **Primario (Azul):** #0066CC
- **Éxito (Verde):** #27AE60
- **Peligro (Rojo):** #E74C3C
- **Neutral (Gris):** #ECF0F1

## Notas

- Los modelos se cargan desde Kedro catalog
- Todas las predicciones se guardan automáticamente en el historial del usuario
- La sesión se mantiene segura con autenticación por contraseña hasheada
- Los tiempos de predicción y probabilidades son simulados en esta versión de demostración
