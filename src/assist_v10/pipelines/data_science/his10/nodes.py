import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

def split_random_data(data: pd.DataFrame, parameters: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Realiza un Split Aleatorio para el modelo de clasificación binaria de inasistencias.
    HIS-10
    """
    test_size = parameters.get("test_size", 0.2)
    random_state = parameters.get("random_state", 42)
    
    # Asumiendo que la variable objetivo es 'no_show' (0: asiste, 1: no asiste)
    if 'no_show' not in data.columns:
        raise ValueError("La columna 'no_show' es requerida para entrenar HIS-10.")
        
    X = data.drop(columns=['no_show'])
    y = data['no_show']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def train_classification_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: Dict[str, Any]) -> CatBoostClassifier:
    """
    Entrena el clasificador CatBoost para propensión de inasistencia (HIS-10).
    Ideal para manejar variables categóricas.
    """
    cat_features = parameters.get('cat_features', [])
    
    # Si faltan características categóricas, autodetectar las de tipo object
    if not cat_features:
        cat_features = list(X_train.select_dtypes(include=['object', 'category']).columns)
        
    model = CatBoostClassifier(
        iterations=parameters.get('iterations', 100),
        learning_rate=parameters.get('learning_rate', 0.1),
        depth=parameters.get('depth', 6),
        cat_features=cat_features,
        verbose=0,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

def evaluate_classification_model(model: CatBoostClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evalúa el modelo HIS-10 usando métricas de clasificación, especialmente pensadas para el 15-25% de ausentismo.
    """
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
    
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    
    return {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "f1_score": float(f1_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
    }
