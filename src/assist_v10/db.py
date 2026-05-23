import sqlite3
import pandas as pd
from datetime import datetime
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "predictions.db")


def init_db():
    """Inicializa la base de datos con las tablas necesarias."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions_his10 (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            username TEXT NOT NULL,
            m_num_exp TEXT,
            med TEXT,
            esp TEXT,
            probabilidad_noshow REAL,
            prediccion_noshow INTEGER
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions_his05 (
            id INTEGER PRIMARY KEY,
            timestamp TEXT NOT NULL,
            username TEXT NOT NULL,
            p_num_exp TEXT,
            p_area TEXT,
            tiempo_estimado_minutos REAL
        )
    """)

    conn.commit()
    conn.close()


def add_user(username: str, password_hash: str):
    """Agrega un usuario a la base de datos."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)",
                      (username, password_hash))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False


def get_user(username: str):
    """Obtiene un usuario de la base de datos."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    result = cursor.fetchone()
    conn.close()
    return result


def save_prediction_his10(username: str, data: dict):
    """Guarda una predicción HIS-10."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions_his10
        (timestamp, username, m_num_exp, med, esp, probabilidad_noshow, prediccion_noshow)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        username,
        data.get("m_num_exp"),
        data.get("med"),
        data.get("esp"),
        data.get("probabilidad_noshow"),
        int(data.get("prediccion_noshow", 0))
    ))
    conn.commit()
    conn.close()


def save_prediction_his05(username: str, data: dict):
    """Guarda una predicción HIS-05."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO predictions_his05
        (timestamp, username, p_num_exp, p_area, tiempo_estimado_minutos)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        username,
        data.get("p_num_exp"),
        data.get("p_area"),
        data.get("tiempo_estimado_minutos")
    ))
    conn.commit()
    conn.close()


def get_predictions_his10(username: str, limit: int = 20):
    """Obtiene historial de predicciones HIS-10."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM predictions_his10 WHERE username = ? ORDER BY timestamp DESC LIMIT ?",
        conn,
        params=(username, limit)
    )
    conn.close()
    return df


def get_predictions_his05(username: str, limit: int = 20):
    """Obtiene historial de predicciones HIS-05."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT * FROM predictions_his05 WHERE username = ? ORDER BY timestamp DESC LIMIT ?",
        conn,
        params=(username, limit)
    )
    conn.close()
    return df
