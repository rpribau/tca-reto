from pydantic import BaseModel, Field
from typing import Optional
from datetime import date, time

# Para HIS-10: Optimizador de Agendas (No-Show Guard) [cite: 17, 19]
class NoShowRequest(BaseModel):
    m_num_exp: str = Field(..., description="Número de expediente (HOSMPI)")
    med: str = Field(..., description="Código del médico (HOSAGD)")
    esp: str = Field(..., description="Especialidad (HOSAGD)")
    a_fecha: date = Field(..., description="Fecha de la cita (HOSAGD)")
    hra_ini: str = Field(..., description="Hora de inicio (HOSAGD)")
    m_cp: str = Field(..., description="Código Postal para cálculo de distancia (HOSMPI)")

class NoShowResponse(BaseModel):
    m_num_exp: str
    probabilidad_noshow: float = Field(..., description="Probabilidad de inasistencia (0.0 a 1.0)")
    prediccion_noshow: bool = Field(..., description="True si se predice que el paciente no asistirá")

# Para HIS-05: Monitor de Tiempos de Espera (NPS AI) [cite: 11, 13]
class WaitTimeRequest(BaseModel):
    p_num_exp: str = Field(..., description="Número de expediente (HOSPAC)")
    p_area: str = Field(..., description="Área actual del hospital (HOSPAC)")
    triage_nivel: int = Field(..., description="Nivel de Triage 1-5 (TRIAGE) [cite: 59, 61]")
    p_fec_lld: date = Field(..., description="Fecha de llegada (HOSPAC)")
    p_hra_lld: str = Field(..., description="Hora de llegada (HOSPAC)")

class WaitTimeResponse(BaseModel):
    p_num_exp: str
    tiempo_estimado_minutos: float = Field(..., description="Tiempo estimado de espera en minutos")
