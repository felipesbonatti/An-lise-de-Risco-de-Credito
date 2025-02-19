from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model, X, y, model_name=None):
    """
    Avalia um modelo com métricas padrão
    
    Args:
        model: Modelo treinado
        X (pd.DataFrame/np.array): Features para avaliação
        y (pd.Series/np.array): Target para avaliação
        model_name (str): Nome do modelo
    
    Returns:
        dict: Dicionário com as métricas de avaliação
    """
    # Previsões
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Métricas
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1': f1_score(y, y_pred),
        'confusion_matrix': confusion_matrix(y, y_pred).tolist()
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
    
    # Log das métricas
    log_prefix = f"{model_name} - " if model_name else ""
    logger.info(f"{log_prefix}Avaliação do modelo:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':
            logger.info(f"{log_prefix}{metric}: {value:.4f}")
    
    return metrics

def save_model(model, model_name, model_dir='./models/trained_models'):
    """
    Salva um modelo treinado
    
    Args:
        model: Modelo treinado
        model_name (str): Nome do modelo
        model_dir (str): Diretório para salvar o modelo
    
    Returns:
        str: Caminho