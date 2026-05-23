import bcrypt
import streamlit as st
from assist_v10.db import get_user, add_user, init_db

init_db()

DEFAULT_USERS = {
    "admin": "admin123",
    "demo": "demo123"
}


def hash_password(password: str) -> str:
    """Genera hash de una contraseña."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Verifica una contraseña contra su hash."""
    return bcrypt.checkpw(password.encode(), password_hash.encode())


def init_default_users():
    """Inicializa usuarios por defecto si no existen."""
    for username, password in DEFAULT_USERS.items():
        if not get_user(username):
            hashed = hash_password(password)
            add_user(username, hashed)


def login():
    """Interfaz de login."""
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: 50px auto;
            padding: 30px;
            border-radius: 10px;
            background-color: #f8f9fa;
            border-left: 4px solid #0066CC;
        }
        .login-title {
            color: #0066CC;
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 10px;
        }
        .login-subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-title">TCA Software Solutions</div>', unsafe_allow_html=True)
        st.markdown('<div class="login-subtitle">Portal de Predicciones Hospitalarias</div>', unsafe_allow_html=True)

        username = st.text_input("Usuario", key="login_user")
        password = st.text_input("Contraseña", type="password", key="login_pass")

        if st.button("Ingresar", key="login_btn", use_container_width=True):
            user_record = get_user(username)
            if user_record and verify_password(password, user_record[0]):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Usuario o contraseña incorrectos")

        st.markdown("---")
        st.info("""
        **Usuarios de prueba:**
        - Usuario: admin | Contraseña: admin123
        - Usuario: demo | Contraseña: demo123
        """)


def check_authentication():
    """Verifica si el usuario está autenticado, sino muestra login."""
    init_default_users()

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login()
        st.stop()
