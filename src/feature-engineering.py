from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def identify_feature_types(df):
    """
    Identifica os tipos de features no DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
    
    Returns:
        tuple: Listas de colunas numéricas e categóricas
    """
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Features numéricas identificadas: {len(numeric_cols)}")
    logger.info(f"Features categóricas identificadas: {len(categorical_cols)}")
    
    return numeric_cols, categorical_cols

def create_preprocessing_pipeline(numeric_cols, categorical_cols, numeric_strategy='median', scaler='standard'):
    """
    Cria um pipeline de pré-processamento para features numéricas e categóricas
    
    Args:
        numeric_cols (list): Lista de colunas numéricas
        categorical_cols (list): Lista de colunas categóricas
        numeric_strategy (str): Estratégia para tratar valores faltantes em colunas numéricas
        scaler (str): Tipo de escalonamento (standard ou minmax)
    
    Returns:
        ColumnTransformer: Pipeline de pré-processamento
    """
    # Pipeline para features numéricas
    if scaler == 'standard':
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=numeric_strategy)),
            ('scaler', StandardScaler())
        ])
    elif scaler == 'minmax':
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=numeric_strategy)),
            ('scaler', MinMaxScaler())
        ])
    else:
        raise ValueError("Scaler deve ser 'standard' ou 'minmax'")
    
    # Pipeline para features categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combinação dos pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    logger.info(f"Pipeline de pré-processamento criado com scaler: {scaler}")
    return preprocessor

def create_new_features(df):
    """
    Cria novas features a partir das existentes
    
    Args:
        df (pd.DataFrame): DataFrame original
    
    Returns:
        pd.DataFrame: DataFrame com novas features
    """
    df_new = df.copy()
    
    # Exemplo: cálculo de razões financeiras
    if 'renda' in df_new.columns and 'divida_total' in df_new.columns:
        df_new['razao_divida_renda'] = df_new['divida_total'] / (df_new['renda'] + 1e-10)
    
    if 'idade' in df_new.columns:
        # Discretização da idade em faixas
        df_new['faixa_etaria'] = pd.cut(
            df_new['idade'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )
    
    if 'historico_credito' in df_new.columns:
        df_new['historico_credito_ruim'] = df_new['historico_credito'].apply(
            lambda x: 1 if x in ['ruim', 'muito_ruim'] else 0
        )
    
    logger.info(f"Novas features criadas. Shape atualizado: {df_new.shape}")
    return df_new

def select_important_features(X_train, y_train, X_val, X_test, method='mutual_info', k=10):
    """
    Seleciona as features mais importantes
    
    Args:
        X_train (pd.DataFrame): Features de treino
        y_train (pd.Series): Target de treino
        X_val (pd.DataFrame): Features de validação
        X_test (pd.DataFrame): Features de teste
        method (str): Método de seleção (mutual_info, chi2, f_classif)
        k (int): Número de features a serem selecionadas
    
    Returns:
        tuple: X_train, X_val, X_test apenas com as features selecionadas
    """
    if method == 'mutual_info':
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        selector = SelectKBest(mutual_info_classif, k=k)
    elif method == 'chi2':
        from sklearn.feature_selection import SelectKBest, chi2
        selector = SelectKBest(chi2, k=k)
    elif method == 'f_classif':
        from sklearn.feature_selection import SelectKBest, f_classif
        selector = SelectKBest(f_classif, k=k)
    else:
        raise ValueError("Método deve ser 'mutual_info', 'chi2' ou 'f_classif'")
    
    # Treina o seletor
    selector.fit(X_train, y_train)
    
    # Obtém os índices das features selecionadas
    selected_indices = selector.get_support(indices=True)
    feature_names = X_train.columns[selected_indices]
    
    # Aplica a seleção
    X_train_selected = X_train.iloc[:, selected_indices]
    X_val_selected = X_val.iloc[:, selected_indices]
    X_test_selected = X_test.iloc[:, selected_indices]
    
    logger.info(f"Features selecionadas ({len(feature_names)}): {feature_names.tolist()}")
    
    return X_train_selected, X_val_selected, X_test_selected, feature_names