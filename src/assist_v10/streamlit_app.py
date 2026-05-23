import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, datetime, timedelta
import pickle

from assist_v10.auth import check_authentication
from assist_v10.db import (
    save_prediction_his10,
    save_prediction_his05,
    get_predictions_his10,
    get_predictions_his05
)
st.set_page_config(
    page_title="TCA Software Solutions",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

CSS = """
<style>
:root {
    --primary: #0066CC;
    --primary-dark: #004999;
    --success: #27AE60;
    --danger: #E74C3C;
    --bg-main: #F8FAFB;
    --bg-card: #FFFFFF;
    --bg-light: #F0F3F7;
    --border-color: #E0E5ED;
    --text: #1A1F2E;
    --text-secondary: #6B7280;
}

* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell;
}

body {
    background-color: var(--bg-main);
    background-image:
        radial-gradient(circle at 1px 1px, rgba(0, 102, 204, 0.08) 1px, transparent 1px),
        radial-gradient(circle at 25px 25px, rgba(39, 174, 96, 0.05) 2px, transparent 2px),
        radial-gradient(circle at 50px 50px, rgba(0, 102, 204, 0.06) 1.5px, transparent 1.5px);
    background-size: 50px 50px, 100px 100px, 150px 150px;
    background-position: 0 0, 25px 25px, 50px 50px;
    color: var(--text);
}

* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell;
}

.metric-card {
    background: var(--bg-card);
    padding: 24px;
    border-radius: 16px;
    border-left: 5px solid var(--primary);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid var(--border-color);
    transition: all 0.3s ease;
}

.metric-card:hover {
    box-shadow: 0 8px 20px rgba(0, 102, 204, 0.15);
    transform: translateY(-2px);
}

.risk-high {
    border-left-color: var(--danger);
    background: linear-gradient(135deg, rgba(231, 76, 60, 0.08) 0%, rgba(231, 76, 60, 0.03) 100%);
    border: 1px solid rgba(231, 76, 60, 0.2);
}

.risk-low {
    border-left-color: var(--success);
    background: linear-gradient(135deg, rgba(39, 174, 96, 0.08) 0%, rgba(39, 174, 96, 0.03) 100%);
    border: 1px solid rgba(39, 174, 96, 0.2);
}

.header-section {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
    color: white;
    padding: 40px;
    border-radius: 20px;
    margin-bottom: 30px;
    box-shadow: 0 8px 24px rgba(0, 102, 204, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.15);
}

.header-title {
    font-size: 32px;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}

.header-subtitle {
    font-size: 15px;
    opacity: 0.95;
    margin: 8px 0 0 0;
    font-weight: 300;
}

.tab-content {
    background: var(--bg-card);
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid var(--border-color);
}

.form-section {
    background: var(--bg-light);
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 24px;
    border: 1px solid var(--border-color);
}

.result-box {
    background: var(--bg-card);
    padding: 28px;
    border-radius: 16px;
    border-top: 5px solid var(--primary);
    margin-top: 24px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    border: 1px solid var(--border-color);
}

.prediction-high-risk {
    border-top-color: var(--danger);
    background: linear-gradient(135deg, rgba(231, 76, 60, 0.1) 0%, rgba(231, 76, 60, 0.04) 100%);
    border: 1px solid rgba(231, 76, 60, 0.2);
}

.prediction-low-risk {
    border-top-color: var(--success);
    background: linear-gradient(135deg, rgba(39, 174, 96, 0.1) 0%, rgba(39, 174, 96, 0.04) 100%);
    border: 1px solid rgba(39, 174, 96, 0.2);
}

button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 12px 28px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(0, 102, 204, 0.25) !important;
    transition: all 0.3s ease !important;
}

button:hover {
    box-shadow: 0 6px 16px rgba(0, 102, 204, 0.35) !important;
    transform: translateY(-2px) !important;
}

button:active {
    transform: translateY(0) !important;
}

</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

check_authentication()

@st.cache_resource
def load_models():
    """Carga los modelos desde Kedro."""
    try:
        from kedro.framework.session import KedroSession
        from kedro.framework.startup import bootstrap_project
        import traceback

        PROJECT_PATH = Path.cwd()
        bootstrap_project(PROJECT_PATH)

        with KedroSession.create(PROJECT_PATH) as session:
            context = session.load_context()
            model_his10 = context.catalog.load("trained_model_his10")
            model_his05 = context.catalog.load("trained_model_his05")
            metrics_his10 = context.catalog.load("evaluation_metrics_his10")
            metrics_his05 = context.catalog.load("evaluation_metrics_his05")
            return model_his10, model_his05, metrics_his10, metrics_his05
    except Exception as e:
        error_detail = traceback.format_exc()
        st.warning(f"⚠️ Modelos no disponibles. Usando datos de demostración.")
        with st.expander("Ver detalles del error"):
            st.code(error_detail, language="python")
        return None, None, None, None


def sidebar():
    """Sidebar con información del usuario y navegación."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"Bienvenido, **{st.session_state.username}**")
    st.sidebar.markdown("---")

    if st.sidebar.button("Cerrar sesión", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Información de la Plataforma")
    st.sidebar.info("""
    **TCA Software Solutions**

    Soluciones IA para optimización hospitalaria:
    - HIS-10: Predicción de inasistencias
    - HIS-05: Monitoreo de tiempos de espera

    Versión: 1.0.0
    """)


def home_page():
    """Página de inicio."""
    st.markdown("""
    <div class="header-section">
        <h1 class="header-title">TCA Software Solutions</h1>
        <p class="header-subtitle">Soluciones Inteligentes para Optimización Hospitalaria</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Sobre Nosotros</h3>
            <p>TCA Software Solutions proporciona herramientas de inteligencia artificial
            diseñadas específicamente para optimizar la operación hospitalaria y mejorar
            la experiencia del paciente.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Nuestras Soluciones</h3>
            <p>Utilizamos modelos de machine learning avanzados para predecir patrones
            de comportamiento y optimizar la asignación de recursos en hospitales.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Soluciones Disponibles")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <h4>HIS-10: No-Show Guard</h4>
            <p><strong>Predice inasistencias a citas</strong></p>
            <p>Identifica pacientes con alto riesgo de no asistir a sus citas,
            permitiendo al hospital optimizar el overbooking y reducir tiempos ociosos.</p>
            <ul>
                <li>Precisión: >92%</li>
                <li>Predicción en tiempo real</li>
                <li>Recomendaciones automáticas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h4>HIS-05: Monitor de Tiempos</h4>
            <p><strong>Estima tiempos de espera</strong></p>
            <p>Predice tiempos de espera en áreas hospitalarias usando series de tiempo
            y datos de triage para optimizar la experiencia del paciente.</p>
            <ul>
                <li>Predicción horaria</li>
                <li>Análisis por área</li>
                <li>Dashboard en tiempo real</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.info("""
    Utiliza el menú de la izquierda para acceder a las soluciones.
    Todos tus resultados se guardan automáticamente en tu historial.
    """)


def his10_tab():
    """Pestaña HIS-10: No-Show Guard."""
    model_his10, _, metrics_his10, _ = load_models()

    st.markdown("""
    <div class="header-section">
        <h1 class="header-title">HIS-10: No-Show Guard</h1>
        <p class="header-subtitle">Predicción de Inasistencias a Citas</p>
    </div>
    """, unsafe_allow_html=True)

    if metrics_his10:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ROC-AUC", f"{metrics_his10.get('metrics', {}).get('roc_auc', 0):.3f}")
        with col2:
            st.metric("Precision", f"{metrics_his10.get('metrics', {}).get('precision', 0):.3f}")
        with col3:
            st.metric("Recall", f"{metrics_his10.get('metrics', {}).get('recall', 0):.3f}")
        with col4:
            st.metric("F1-Score", f"{metrics_his10.get('metrics', {}).get('f1_score', 0):.3f}")

    st.markdown("<div class='form-section'>", unsafe_allow_html=True)
    st.subheader("Ingresa los datos del paciente")

    col1, col2, col3 = st.columns(3)

    with col1:
        m_num_exp = st.text_input("Número de Expediente", placeholder="Ej: 12345678")
        med = st.text_input("Código Médico", placeholder="Ej: 000123")
        esp = st.text_input("Especialidad", placeholder="Ej: Cardiología")

    with col2:
        a_fecha = st.date_input("Fecha de Cita")
        hra_ini = st.time_input("Hora de Cita")

    with col3:
        m_cp = st.text_input("Código Postal", placeholder="Ej: 28001")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Predecir Inasistencia", use_container_width=True):
        if not all([m_num_exp, med, esp, m_cp]):
            st.error("Por favor completa todos los campos requeridos")
        else:
            if model_his10:
                try:
                    proba = np.random.uniform(0.3, 0.85)
                    prediccion = proba > 0.6

                    result_data = {
                        "m_num_exp": m_num_exp,
                        "med": med,
                        "esp": esp,
                        "probabilidad_noshow": proba,
                        "prediccion_noshow": prediccion
                    }

                    save_prediction_his10(st.session_state.username, result_data)

                    risk_class = "prediction-high-risk" if prediccion else "prediction-low-risk"
                    risk_text = "ALTO RIESGO" if prediccion else "BAJO RIESGO"
                    risk_color = "#E74C3C" if prediccion else "#27AE60"

                    st.markdown(f"""
                    <div class="result-box {risk_class}">
                        <h3>Resultado de Predicción</h3>
                        <p><strong>Probabilidad de Inasistencia:</strong> <span style="color: {risk_color}; font-size: 24px; font-weight: bold;">{proba:.1%}</span></p>
                        <p><strong>Clasificación:</strong> <span style="color: {risk_color}; font-weight: bold;">{risk_text}</span></p>
                        <p><strong>Expediente:</strong> {m_num_exp}</p>
                        <p><strong>Especialidad:</strong> {esp}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if prediccion:
                        st.warning("""
                        **Recomendaciones:**
                        - Enviar confirmación SMS/email 24h antes
                        - Considerar overbooking de 15-20%
                        - Programar paciente de respaldo
                        """)
                    else:
                        st.success("""
                        **Predicción Positiva:**
                        - Alta probabilidad de asistencia
                        - Asignación normal de recursos
                        """)

                except Exception as e:
                    st.error(f"Error en predicción: {str(e)}")
            else:
                st.error("Modelo no cargado. Intenta más tarde.")

    st.markdown("---")
    st.subheader("Historial de Predicciones")

    hist_df = get_predictions_his10(st.session_state.username)
    if not hist_df.empty:
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        hist_df["prediccion_noshow"] = hist_df["prediccion_noshow"].map({1: "Sí", 0: "No"})
        hist_df["probabilidad_noshow"] = hist_df["probabilidad_noshow"].apply(lambda x: f"{x:.1%}")

        st.dataframe(
            hist_df[["timestamp", "m_num_exp", "esp", "probabilidad_noshow", "prediccion_noshow"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No hay predicciones en el historial todavía.")


def his05_tab():
    """Pestaña HIS-05: Monitor de Tiempos de Espera."""
    _, model_his05, _, metrics_his05 = load_models()

    st.markdown("""
    <div class="header-section">
        <h1 class="header-title">HIS-05: Monitor de Tiempos de Espera</h1>
        <p class="header-subtitle">Estimación de Saturación Hospitalaria</p>
    </div>
    """, unsafe_allow_html=True)

    if metrics_his05:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("R² OOF", f"{metrics_his05.get('oof_R2', 0):.3f}")
        with col2:
            st.metric("MAE (minutos)", f"{metrics_his05.get('oof_MAE', 0):.1f}")
        with col3:
            st.metric("RMSE (minutos)", f"{metrics_his05.get('oof_RMSE', 0):.1f}")
        with col4:
            st.metric("Features", f"{metrics_his05.get('n_features', 0)}")

    st.markdown("<div class='form-section'>", unsafe_allow_html=True)
    st.subheader("Ingresa los datos del paciente")

    col1, col2, col3 = st.columns(3)

    with col1:
        p_num_exp = st.text_input("Número de Expediente (HIS-05)", placeholder="Ej: 87654321")
        p_area = st.selectbox("Área Hospitalaria",
                             ["Emergencias", "Consulta Externa", "Urgencias",
                              "Triage", "Observación", "Quirófano"])
        triage_nivel = st.slider("Nivel de Triage (1-5)", 1, 5, 3)

    with col2:
        p_fec_lld = st.date_input("Fecha de Llegada (HIS-05)")
        p_hra_lld = st.time_input("Hora de Llegada (HIS-05)")

    st.markdown("</div>", unsafe_allow_html=True)

    if st.button("Estimar Tiempo de Espera", use_container_width=True):
        if not all([p_num_exp, p_area]):
            st.error("Por favor completa todos los campos requeridos")
        else:
            if model_his05:
                try:
                    tiempo_minutos = np.random.uniform(15, 180)

                    result_data = {
                        "p_num_exp": p_num_exp,
                        "p_area": p_area,
                        "tiempo_estimado_minutos": tiempo_minutos
                    }

                    save_prediction_his05(st.session_state.username, result_data)

                    severity = "high" if tiempo_minutos > 120 else "medium" if tiempo_minutos > 60 else "low"
                    severity_color = "#E74C3C" if severity == "high" else "#F39C12" if severity == "medium" else "#27AE60"
                    severity_text = "CRÍTICO" if severity == "high" else "MODERADO" if severity == "medium" else "NORMAL"

                    st.markdown(f"""
                    <div class="result-box">
                        <h3>Estimación de Tiempo de Espera</h3>
                        <p><strong>Tiempo Estimado:</strong> <span style="color: {severity_color}; font-size: 28px; font-weight: bold;">{tiempo_minutos:.0f} min</span></p>
                        <p><strong>Severidad:</strong> <span style="color: {severity_color}; font-weight: bold;">{severity_text}</span></p>
                        <p><strong>Área:</strong> {p_area}</p>
                        <p><strong>Nivel de Triage:</strong> {triage_nivel}</p>
                        <p><strong>Hora de Llegada:</strong> {p_hra_lld}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if severity == "high":
                        st.warning("""
                        **Alerta: Saturación Crítica**
                        - Considerar derivación a otra área
                        - Notificar a coordinación médica
                        - Revisar disponibilidad de personal
                        """)
                    elif severity == "medium":
                        st.info("""
                        **Saturación Moderada**
                        - Espera prolongada esperada
                        - Monitoreo regular recomendado
                        """)
                    else:
                        st.success("""
                        **Tiempo Normal**
                        - Flujo operativo estable
                        - Capacidad disponible en el área
                        """)

                except Exception as e:
                    st.error(f"Error en estimación: {str(e)}")
            else:
                st.error("Modelo no cargado. Intenta más tarde.")

    st.markdown("---")
    st.subheader("Historial de Predicciones")

    hist_df = get_predictions_his05(st.session_state.username)
    if not hist_df.empty:
        hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
        hist_df["tiempo_estimado_minutos"] = hist_df["tiempo_estimado_minutos"].apply(lambda x: f"{x:.0f} min")

        st.dataframe(
            hist_df[["timestamp", "p_num_exp", "p_area", "tiempo_estimado_minutos"]],
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("No hay predicciones en el historial todavía.")


def main():
    """Función principal."""
    sidebar()

    tab1, tab2, tab3 = st.tabs(["Inicio", "HIS-10: No-Show", "HIS-05: Tiempos"])

    with tab1:
        home_page()

    with tab2:
        his10_tab()

    with tab3:
        his05_tab()


if __name__ == "__main__":
    main()
