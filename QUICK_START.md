# TCA Software Solutions - Guía Rápida de Inicio

## Requisitos Previos

- Python 3.8+
- Git
- Pip

## Pasos de Instalación

### 1. Instalar Dependencias

```bash
pip install -r requirements.txt
```

Este comando instalará:
- Streamlit (interfaz web)
- Plotly (gráficos interactivos)
- SQLAlchemy y bcrypt (base de datos y autenticación)
- Todos los modelos y dependencias de ML

### 2. Ejecutar la Aplicación

**En Windows:**
```bash
run_streamlit.bat
```

**En Linux/Mac:**
```bash
bash run_streamlit.sh
```

**O directamente:**
```bash
streamlit run src/assist_v10/streamlit_app.py
```

### 3. Acceder a la Interfaz

La aplicación se abrirá automáticamente en tu navegador:
- **URL:** http://localhost:8501

## Inicio de Sesión

Usa cualquiera de estas credenciales de prueba:

| Usuario | Contraseña | Rol |
|---------|-----------|-----|
| admin   | admin123  | Administrador |
| demo    | demo123   | Usuario Demo |

## Características Principales

### Inicio
- Información sobre la plataforma TCA
- Descripción de soluciones
- Acceso a ambos modelos

### HIS-10: No-Show Guard
- Predicción de inasistencias a citas médicas
- Probabilidad en tiempo real
- Recomendaciones automáticas
- Historial de predicciones

### HIS-05: Monitor de Tiempos de Espera
- Estimación de saturación hospitalaria
- Análisis por área y triage
- Alertas de ocupación crítica
- Registro completo de consultas

## Archivos Importantes

```
tca-reto/
├── run_streamlit.bat              # Script de inicio (Windows)
├── run_streamlit.sh               # Script de inicio (Linux/Mac)
├── requirements.txt               # Dependencias
├── STREAMLIT_README.md            # Documentación completa
├── QUICK_START.md                 # Este archivo
├── .streamlit/
│   └── config.toml               # Configuración Streamlit
└── src/assist_v10/
    ├── streamlit_app.py          # App principal
    ├── auth.py                   # Autenticación
    ├── db.py                     # Base de datos
    └── predictions.db            # (Se crea automáticamente)
```

## Estructura de la Base de Datos

La aplicación crea automáticamente `src/assist_v10/predictions.db` con:

- **users:** Registro de usuarios y contraseñas hasheadas
- **predictions_his10:** Historial de predicciones de no-show
- **predictions_his05:** Historial de tiempos de espera

## Diseño de Color

La interfaz usa una paleta profesional para hospitales:

- **Azul #0066CC:** Color primario, información
- **Verde #27AE60:** Indicadores positivos, bajo riesgo
- **Rojo #E74C3C:** Alertas, alto riesgo
- **Gris #ECF0F1:** Fondo, elementos neutrales

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install streamlit -U
```

### Error: "Port 8501 already in use"
```bash
streamlit run src/assist_v10/streamlit_app.py --server.port 8502
```

### La base de datos no se crea
- Verifica permisos de escritura en el directorio `src/assist_v10/`
- Intenta ejecutar: `python -c "from assist_v10.db import init_db; init_db()"`

## Contacto

Para reportar problemas o sugerencias:
- Email: soporte@tcasoftware.com
- Documentación: Ver STREAMLIT_README.md

---

**Versión:** 1.0.0
**Última actualización:** 2026-05-22
