import pandas as pd
from feature_engineering_train import get_period_day, is_high_season, get_min_diff

DATA_PATH = "../data/data.csv"
TRAIN_PATH = "../data/data_train.csv"
OUTPUT_INF_PATH = "../data/data_inf.csv"

def preprocess_inference(data: pd.DataFrame, train_columns: list) -> pd.DataFrame:
    """
    Preprocesses the inference dataset to match training dataset features.

    Args:
        data (pd.DataFrame): Raw inference data.
        train_columns (list): List of expected feature columns based on training data.

    Returns:
        pd.DataFrame: Processed inference dataset.
    """
    data['period_day'] = data['Fecha-I'].apply(get_period_day)
    data['high_season'] = data['Fecha-I'].apply(is_high_season)
    data['min_diff'] = data.apply(get_min_diff, axis=1)

    features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix='OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
        pd.get_dummies(data['MES'], prefix='MES')
    ], axis=1)

    # Asegurar que las columnas coincidan con las del conjunto de entrenamiento
    features = features.reindex(columns=train_columns, fill_value=0)

    return features

if __name__ == "__main__":
    # Cargar datos completos y dividir en inferencia (20%)
    data = pd.read_csv(DATA_PATH)
    data_inf = data.sample(frac=0.2, random_state=42).reset_index(drop=True)

    # Cargar columnas esperadas del conjunto de entrenamiento
    train_data = pd.read_csv(TRAIN_PATH)
    train_columns = train_data.drop(columns=['delay']).columns.tolist()

    # Aplicar feature engineering asegurando que las columnas coincidan
    processed_inf = preprocess_inference(data_inf, train_columns)

    # Guardar el conjunto de inferencia
    processed_inf.to_csv(OUTPUT_INF_PATH, index=False)
    print(f"✅ Archivo guardado en {OUTPUT_INF_PATH}")
