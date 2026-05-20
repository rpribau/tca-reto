from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import pandas as pd
from assist_v10.api.schemas import NoShowRequest, WaitTimeRequest, NoShowResponse, WaitTimeResponse, SimulationRequest
from assist_v10.api.kpi_service import (
    get_kpi_summary,
    get_noshow_rate,
    get_noshow_by_area,
    get_noshow_by_month,
    get_wait_time_estimate,
    get_utilization,
    get_satisfaction_index,
    get_model_performance,
    get_optuna_trials,
    simulate_business_impact,
    get_staffing_recommendations,
)
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

# Servir archivos estáticos del dashboard
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── Dashboard ────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def dashboard():
    """Serve the main dashboard page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Dashboard not found. Place index.html in api/static/"}


# ─── Health ───────────────────────────────────────────────────────

@app.get("/v1/health")
async def health_check():
    """Endpoint para monitoreo de salud en Azure."""
    return {"status": "healthy", "service": "assist_v10_ai"}


# ─── KPI Endpoints ───────────────────────────────────────────────

@app.get("/v1/kpis/summary")
async def kpi_summary():
    """Return all 4 main KPIs in a single response."""
    try:
        return get_kpi_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/kpis/noshow-rate")
async def kpi_noshow_rate():
    """No-show rate with full breakdown."""
    try:
        return get_noshow_rate()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/kpis/noshow-by-area")
async def kpi_noshow_by_area():
    """No-show rate broken down by hospital area."""
    try:
        return get_noshow_by_area()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/kpis/noshow-by-month")
async def kpi_noshow_by_month():
    """No-show rate trend by month."""
    try:
        return get_noshow_by_month()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/kpis/wait-time")
async def kpi_wait_time():
    """Estimated wait time statistics."""
    try:
        return get_wait_time_estimate()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/kpis/utilization")
async def kpi_utilization():
    """Office utilization rate."""
    try:
        return get_utilization()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/kpis/satisfaction")
async def kpi_satisfaction():
    """Composite satisfaction index."""
    try:
        return get_satisfaction_index()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/kpis/model-performance")
async def kpi_model_performance():
    """Model evaluation metrics for HIS-10 and HIS-05."""
    try:
        return get_model_performance()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/kpis/optuna-trials")
async def kpi_optuna_trials():
    """Optuna hyperparameter search results."""
    try:
        trials = get_optuna_trials()
        if trials is None:
            return {"trials": [], "message": "No trial data found"}
        return {"trials": trials}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/kpis/simulate")
async def kpi_simulate(request: SimulationRequest):
    """Run real-time business and financial simulation based on input metrics."""
    try:
        return simulate_business_impact(
            threshold=request.threshold,
            overbooking_rate=request.overbooking_rate,
            consultation_cost=request.consultation_cost,
            hourly_overtime_cost=request.hourly_overtime_cost
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/kpis/staffing")
async def kpi_staffing():
    """Generate staffing recommendations based on HIS-05 demand forecast peaks."""
    try:
        return get_staffing_recommendations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Prediction Endpoints ────────────────────────────────────────


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