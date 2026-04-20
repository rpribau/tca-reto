import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
import xgboost as xgb

def split_time_series_data(data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Realiza un Time Series Split para el modelo de previsión de la saturación.
    HIS-05
    """
    tscv = TimeSeriesSplit(n_splits=parameters.get('n_splits', 5))
    
    # Asumiendo que 'data' ya está ordenado temporalmente y procesado
    # y la variable a predecir es 'tiempo_espera'
    if 'tiempo_espera' not in data.columns:
        raise ValueError("La columna 'tiempo_espera' es requerida para entrenar HIS-05.")
        
    X = data.drop(columns=['tiempo_espera'])
    y = data['tiempo_espera']
    
    # Tomaremos el último split como train/test para devolver
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    return X_train, X_test, y_train, y_test

def train_forecasting_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict[str, Any]) -> XGBRegressor:
    """
    Entrena el regresor XGBoost para el forecasting de saturación (HIS-05).
    """
    model = XGBRegressor(
        n_estimators=parameters.get('n_estimators', 100),
        learning_rate=parameters.get('learning_rate', 0.1),
        max_depth=parameters.get('max_depth', 5),
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_forecasting_model(model: XGBRegressor, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evalúa el modelo HIS-05 y devuelve métricas.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    
    return {
        "mae": float(mae),
        "mse": float(mse)
    }
