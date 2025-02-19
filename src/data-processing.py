import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data_from_database(db_connection_string, query):
    """
    Carrega dados de um banco de dados SQL
    
    Args:
        db_connection_string (str): String de conexão com o banco de dados
        query (str): Query SQL para extrair os dados
        
    Returns:
        pd.DataFrame: Dados carregados do banco de dados
    """
    try:
        engine = create_engine(db_connection_string)
        logger.info("Conectado ao banco de dados com sucesso")
        df = pd.read_sql(query, engine)
        logger.info(f"Dados carregados com sucesso. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar dados do banco: {str(e)}")
        raise

def load_data_from_csv(file_path):
    """
    Carrega dados de um arquivo CSV
    
    Args:
        file_path (str): Caminho para o arquivo CSV
        
    Returns:
        pd.DataFrame: Dados carregados do arquivo CSV
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dados carregados com sucesso do arquivo {file_path}. Shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Erro ao carregar dados do arquivo CSV: {str(e)}")
        raise

def handle_missing_values(df, strategy='median'):
    """
    Trata valores faltantes no DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame com valores faltantes
        strategy (str): Estratégia para tratar valores faltantes (median, mean, mode, drop)
        
    Returns:
        pd.DataFrame: DataFrame sem valores faltantes
    """
    df_copy = df.copy()
    
    # Verifica se há valores faltantes
    missing_values = df_copy.isnull().sum()
    if missing_values.sum() == 0:
        logger.info("Não há valores faltantes no DataFrame")
        return df_copy
    
    logger.info(f"Valores faltantes encontrados: \n{missing_values[missing_values > 0]}")
    
    # Trata valores faltantes de acordo com a estratégia
    for col in df_copy.columns:
        if df_copy[col].isnull().sum() > 0:
            if strategy == 'median' and df_copy[col].dtype in ['int64', 'float64']:
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())
            elif strategy == 'mean' and df_copy[col].dtype in ['int64', 'float64']:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mean())
            elif strategy == 'mode':
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])
            elif strategy == 'drop':
                df_copy = df_copy.dropna(subset=[col])
            else:
                logger.warning(f"Estratégia {strategy} não aplicável para a coluna {col}")
    
    logger.info(f"Valores faltantes tratados com a estratégia: {strategy}")
    return df_copy

def handle_outliers(df, method='iqr', columns=None):
    """
    Trata outliers no DataFrame
    
    Args:
        df (pd.DataFrame): DataFrame com outliers
        method (str): Método para tratar outliers (iqr, zscore)
        columns (list): Lista de colunas para tratar outliers (se None, aplicado em todas as colunas numéricas)
        
    Returns:
        pd.DataFrame: DataFrame sem outliers
    """
    df_copy = df.copy()
    
    if columns is None:
        columns = df_copy.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    for col in columns:
        if method == 'iqr':
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Substitui outliers pelos limites
            df_copy.loc[df_copy[col] < lower_bound, col] = lower_bound
            df_copy.loc[df_copy[col] > upper_bound, col] = upper_bound
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = stats.zscore(df_copy[col])
            abs_z_scores = np.abs(z_scores)
            filtered_entries = (abs_z_scores < 3)
            df_copy.loc[~filtered_entries, col] = df_copy[col].mean()
    
    logger.info(f"Outliers tratados com o método: {method}")
    return df_copy

def split_data(df, target_column, test_size=0.2, validation_size=0.2, random_state=42):
    """
    Divide os dados em conjuntos de treino, validação e teste
    
    Args:
        df (pd.DataFrame): DataFrame com os dados
        target_column (str): Nome da coluna alvo
        test_size (float): Proporção do conjunto de teste
        validation_size (float): Proporção do conjunto de validação
        random_state (int): Seed para reprodutibilidade
        
    Returns:
        tuple: X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Primeiro, divide em treino e teste
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Depois, divide o conjunto de treino em treino e validação
    val_size_adjusted = validation_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, 
        random_state=random_state, stratify=y_train_val
    )
    
    logger.info(f"Dados divididos - Treino: {X_train.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test