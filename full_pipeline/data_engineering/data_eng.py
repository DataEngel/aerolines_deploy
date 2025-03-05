import pandas as pd
from datetime import datetime
from google.cloud import storage
import io

# Configuración de rutas en Google Cloud Storage
BUCKET_NAME = "testoneml"
DATA_PATH = "latam-model/data-dirt/data.csv"
OUTPUT_TRAIN_PATH = "latam-model/training/data-post-feature-eng/data_train.csv"
OUTPUT_INF_PATH = "latam-model/inference/data-post-feature-eng/data_inf.csv"

def download_csv_from_gcs(bucket_name: str, file_path: str) -> pd.DataFrame:
    """Descarga un archivo CSV desde Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    data = blob.download_as_bytes()
    return pd.read_csv(io.BytesIO(data))

def upload_csv_to_gcs(bucket_name: str, file_path: str, df: pd.DataFrame):
    """Sube un DataFrame como archivo CSV a Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)
    blob.upload_from_string(df.to_csv(index=False), 'text/csv')
    print(f"✅ Archivo guardado en gs://{bucket_name}/{file_path}")

def get_period_day(date: str) -> str:
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    if datetime.strptime("05:00", '%H:%M').time() <= date_time <= datetime.strptime("11:59", '%H:%M').time():
        return 'mañana'
    elif datetime.strptime("12:00", '%H:%M').time() <= date_time <= datetime.strptime("18:59", '%H:%M').time():
        return 'tarde'
    else:
        return 'noche'

def is_high_season(fecha: str) -> int:
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    ranges = [
        (datetime.strptime('15-Dec', '%d-%b'), datetime.strptime('31-Dec', '%d-%b')),
        (datetime.strptime('1-Jan', '%d-%b'), datetime.strptime('3-Mar', '%d-%b')),
        (datetime.strptime('15-Jul', '%d-%b'), datetime.strptime('31-Jul', '%d-%b')),
        (datetime.strptime('11-Sep', '%d-%b'), datetime.strptime('30-Sep', '%d-%b'))
    ]
    for r_min, r_max in ranges:
        if r_min.replace(year=fecha_año) <= fecha <= r_max.replace(year=fecha_año):
            return 1
    return 0

def get_min_diff(row: pd.Series) -> float:
    fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    return (fecha_o - fecha_i).total_seconds() / 60

def preprocess_data(data: pd.DataFrame, train_columns=None) -> pd.DataFrame:
    """
    Aplica Feature Engineering a los datos y asegura que coincidan con el formato del conjunto de entrenamiento.

    Args:
        data (pd.DataFrame): Datos crudos.
        train_columns (list, optional): Columnas esperadas para el conjunto de inferencia.

    Returns:
        pd.DataFrame: Datos con Feature Engineering aplicado.
    """
    data['period_day'] = data['Fecha-I'].apply(get_period_day)
    data['high_season'] = data['Fecha-I'].apply(is_high_season)
    data['min_diff'] = data.apply(get_min_diff, axis=1)

    # Variables categóricas
    features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix='OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
        pd.get_dummies(data['MES'], prefix='MES')
    ], axis=1)

    # Si estamos procesando el conjunto de entrenamiento, agregamos la variable objetivo
    if train_columns is None:
        data['delay'] = (data['min_diff'] > 15).astype(int)
        features['delay'] = data['delay']
        return features
    else:
        # Para inferencia: asegurar que las columnas coincidan con las de entrenamiento
        return features.reindex(columns=train_columns, fill_value=0)

if __name__ == "__main__":
    print(f"📥 Cargando dataset desde gs://{BUCKET_NAME}/{DATA_PATH} ...")
    data = download_csv_from_gcs(BUCKET_NAME, DATA_PATH)

    # Dividir datos en entrenamiento (80%) e inferencia (20%)
    data_train = data.sample(frac=0.8, random_state=42).reset_index(drop=True)
    data_inf = data.drop(data_train.index).reset_index(drop=True)

    print("⚙️ Aplicando Feature Engineering...")
    processed_train = preprocess_data(data_train)

    # Guardar columnas del conjunto de entrenamiento para usarlas en inferencia
    train_columns = processed_train.columns.tolist()

    # Procesar conjunto de inferencia asegurando que tenga las mismas columnas
    processed_inf = preprocess_data(data_inf, train_columns=train_columns)

    print(f"📤 Guardando dataset de entrenamiento en gs://{BUCKET_NAME}/{OUTPUT_TRAIN_PATH} ...")
    upload_csv_to_gcs(BUCKET_NAME, OUTPUT_TRAIN_PATH, processed_train)

    print(f"📤 Guardando dataset de inferencia en gs://{BUCKET_NAME}/{OUTPUT_INF_PATH} ...")
    upload_csv_to_gcs(BUCKET_NAME, OUTPUT_INF_PATH, processed_inf)
