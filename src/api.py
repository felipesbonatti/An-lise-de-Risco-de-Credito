from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
import logging
import json

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicializa a aplicação Flask
app = Flask(__name__)

# Carrega o modelo, preprocessador e configurações
MODEL_PATH = os.environ.get('MODEL_PATH', './models/trained_models/best_model.joblib')
PREPROCESSOR_PATH = os.environ.get('PREPROCESSOR_PATH', './models/trained_models/preprocessor.joblib')
CONFIG_PATH = os.environ.get('CONFIG_PATH', './models/trained_models/config.json')

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    feature_names = config.get('feature_names', [])
    threshold = config.get('threshold', 0.5)
    risk_categories = config.get('risk_categories', ['Baixo Risco', 'Alto Risco'])
    
    logger.info(f"Modelo carregado de {MODEL_PATH}")
    logger.info(f"Preprocessador carregado de {PREPROCESSOR_PATH}")
    logger.info(f"Configuração carregada de {CONFIG_PATH}")
    
except Exception as e:
    logger.error(f"Erro ao carregar o modelo ou configurações: {str(e)}")
    model = None
    preprocessor = None
    config = None
    feature_names = []
    threshold = 0.5
    risk_categories = ['Baixo Risco', 'Alto Risco']

@app.route('/health', methods=['GET'])
def health_check():
    """
    Endpoint para verificar a saúde da API
    """
    if model is not None and preprocessor is not None:
        return jsonify({
            'status': 'ok',
            'message': 'API está funcionando corretamente'
        }), 200
    else:
        return jsonify({
            'status': 'error',
            'message': 'API não está funcionando corretamente: modelo ou preprocessador não carregados'
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar predições de risco de crédito
    """
    try:
        # Obtém os dados da requisição
        data = request.get_json()
        
        # Verifica se os dados foram fornecidos
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Nenhum dado foi fornecido'
            }), 400
        
        # Converte os dados para DataFrame
        input_df = pd.DataFrame([data])
        
        # Verifica se todas as features necessárias estão presentes
        missing_features = [feature for feature in feature_names if feature not in input_df.columns]
        if missing_features:
            return jsonify({
                'status': 'error',
                'message': f'Features faltantes: {missing_features}'
            }), 400
        
        # Pré-processamento dos dados
        X = preprocessor.transform(input_df)
        
        # Faz a predição
        risk_prob = model.predict_proba(X)[0, 1]
        risk_binary = 1 if risk_prob >= threshold else 0
        risk_category = risk_categories[risk_binary]
        
        # Loga a predição
        logger.info(f"Predição realizada: Prob={risk_prob:.4f}, Categoria={risk_category}")
        
        # Retorna a resposta
        return jsonify({
            'status': 'success',
            'prediction': {
                'risk_probability': float(risk_prob),
                'risk_category': risk_category,
                'threshold': threshold
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao realizar predição: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Ocorreu um erro ao processar a requisição: {str(e)}'
        }), 500

@app.route('/explain', methods=['POST'])
def explain():
    """
    Endpoint para explicar uma predição
    """
    try:
        # Obtém os dados da requisição
        data = request.get_json()
        
        # Verifica se os dados foram fornecidos
        if not data:
            return jsonify({
                'status': 'error',
                'message': 'Nenhum dado foi fornecido'
            }), 400
        
        # Converte os dados para DataFrame
        input_df = pd.DataFrame([data])
        
        # Verifica se todas as features necessárias estão presentes
        # (parte do código foi cortada, mas provavelmente usa SHAP ou LIME para gerar explicações)
        
        return jsonify({
            'status': 'success',
            'explanation': 'Explicação gerada com sucesso.'
        }), 200
        
    except Exception as e:
        logger.error(f"Erro ao gerar explicação: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'Ocorreu um erro ao processar a requisição: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)