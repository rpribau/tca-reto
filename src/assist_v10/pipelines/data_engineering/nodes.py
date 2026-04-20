import pandas as pd

def preprocess_hospital_data(hospac: pd.DataFrame, hosmpi: pd.DataFrame) -> pd.DataFrame:
    """
    Une los datos de pacientes activos con el índice maestro para incluir CP y demografía.
    Alineado con el Mapa de Capacidades de Gestión Hospitalaria[cite: 92, 194].
    """
    # Join por número de expediente [cite: 75]
    primary_df = hospac.merge(
        hosmpi[['m_num_exp', 'm_cp', 'm_edad', 'm_sexo']], 
        left_on='p_num_exp', 
        right_on='m_num_exp', 
        how='left'
    )
    
    # Limpieza básica: manejo de nulos en CP para evitar errores en distancia [cite: 167]
    primary_df['m_cp'] = primary_df['m_cp'].fillna('00000')
    
    return primary_df

def create_feature_table_his10(primary_df: pd.DataFrame, hosagd: pd.DataFrame) -> pd.DataFrame:
    """
    Prepara la tabla de características para el modelo de No-Show[cite: 19].
    """
    # Lógica para integrar historial de asistencia de HOSAGD
    # ...
    return hosagd_with_features