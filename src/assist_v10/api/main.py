from fastapi import FastAPI, HTTPException
from pathlib import Path
import pandas as pd
from assist_v10.api.schemas import NoShowRequest, WaitTimeRequest, NoShowResponse, WaitTimeResponse
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project

# Inicialización de la aplicación
app = FastAPI(
    title="Assist v10 AI API",
    description="Servicios predictivos para gestión hospitalaria y optimización de agendas",
    version="1.0.0"
)

# Configuración de la ruta del proyecto para Kedro
PROJECT_PATH = Path.cwd()
bootstrap_project(PROJECT_PATH)

@app.get("/v1/health")
async def health_check():
    """Endpoint para monitoreo de salud en Azure."""
    return {"status": "healthy", "service": "assist_v10_ai"}

@app.post("/v1/predict/wait-time", response_model=WaitTimeResponse)
async def predict_wait_time(request: WaitTimeRequest):
    """
    HIS-05: Monitor de Tiempos de Espera (NPS AI).
    Carga el modelo de series de tiempo y devuelve la proyección de saturación.
    """
    try:
        with KedroSession.create(PROJECT_PATH) as session:
            context = session.load_context()
            # Se carga el modelo registrado en el catalog.yml
            model = context.catalog.load("trained_model_his05")
            
            # Convertimos el request a un DataFrame para el modelo
            input_data = pd.DataFrame([request.dict()])
            
            # Esqueleto de predicción
            # prediction = model.predict(input_data)
            prediction_value = 45.0  # Valor simulado basado en el Business Case
            
            return WaitTimeResponse(
                p_num_exp=request.p_num_exp,
                tiempo_estimado_minutos=prediction_value
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia HIS-05: {str(e)}")

@app.post("/v1/predict/no-show", response_model=NoShowResponse)
async def predict_no_show(request: NoShowRequest):
    """
    HIS-10: Optimizador de Agendas (No-Show Guard).
    Predice la probabilidad de inasistencia basándose en el historial y perfil.
    """
    try:
        with KedroSession.create(PROJECT_PATH) as session:
            context = session.load_context()
            # Se carga el clasificador registrado en el catalog.yml
            model = context.catalog.load("trained_model_his10")
            
            # Convertimos el request a DataFrame
            input_data = pd.DataFrame([request.dict()])
            
            # Esqueleto de predicción de probabilidad
            # proba = model.predict_proba(input_data)[:, 1]
            proba_value = 0.72  # Valor simulado del Business Case
            
            return NoShowResponse(
                m_num_exp=request.m_num_exp,
                probabilidad_noshow=proba_value,
                prediccion_noshow=proba_value > 0.6
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en inferencia HIS-10: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)