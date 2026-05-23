#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:src"

echo "Iniciando TCA Software Solutions - Streamlit App"
echo "==============================================="
echo ""
echo "URL local: http://localhost:8501"
echo "Presiona Ctrl+C para detener la aplicación"
echo ""

cd "$(dirname "$0")"
python -m streamlit run src/assist_v10/streamlit_app.py
