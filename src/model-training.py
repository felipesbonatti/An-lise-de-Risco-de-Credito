from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_decision_tree(X_train, y_train, **kwargs):
    """
    Treina um modelo de Árvore de Decisão
    
    Args:
        X_train (pd.DataFrame/np.array): Features de treino
        y_train (pd.Series/np.array): Target de treino
        **kwargs: Parâmetros do modelo
    
    Returns:
        DecisionTreeClassifier: Modelo treinado
    """
    # Parâmetros padrão
    params = {
        'max_depth': 5,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }
    
    # Atualiza parâmetros com os fornecidos
    params.update(kwargs)
    
    # Cria e treina o modelo
    model = DecisionTreeClassifier(**params)
    model.fit(X_train, y_train)
    
    logger.info(f"Modelo de Árvore de Decisão treinado com parâmetros: {params}")
    return model

def train_logistic_regression(X_train, y_train, **kwargs):
    """
    Treina um modelo de Regressão Logística
    
    Args:
        X_train (pd.DataFrame/np.array): Features de treino
        y_train (pd.Series/np.array): Target de treino
        **kwargs: Parâmetros do modelo
    
    Returns:
        LogisticRegression: Modelo treinado
    """
    # Parâmetros padrão
    params = {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_state': 42
    }
    
    # Atualiza parâmetros com os fornecidos
    params.update(kwargs)
    
    # Cria e treina o modelo
    model = LogisticRegression(**params)
    model.fit(X_train, y_train)
    
    logger.info(f"Modelo de Regressão Logística treinado com parâmetros: {params}")
    return model

def train_random_forest(X_train, y_train, **kwargs):
    """
    Treina um modelo Random Forest
    
    Args:
        X_train (pd.DataFrame/np.array): Features de treino
        y_train (pd.Series/np.array): Target de treino
        **kwargs: Parâmetros do modelo
    
    Returns:
        RandomForestClassifier: Modelo treinado
    """
    # Parâmetros padrão
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'random_state': 42
    }
    
    # Atualiza parâmetros com os fornecidos
    params.update(kwargs)
    
    # Cria e treina o modelo
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    logger.info(f"Modelo Random Forest treinado com parâmetros: {params}")
    return model

def train_xgboost(X_train, y_train, **kwargs):
    """
    Treina um modelo XGBoost
    
    Args:
        X_train (pd.DataFrame/np.array): Features de treino
        y_train (pd.Series/np.array): Target de treino
        **kwargs: Parâmetros do modelo
    
    Returns:
        xgb.XGBClassifier: Modelo treinado
    """
    # Parâmetros padrão
    params = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42
    }
    
    # Atualiza parâmetros com os fornecidos
    params.update(kwargs)
    
    # Cria e treina o modelo
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    logger.info(f"Modelo XGBoost treinado com parâmetros: {params}")
    return model