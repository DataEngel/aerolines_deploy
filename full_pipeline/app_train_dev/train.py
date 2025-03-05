import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from google.cloud import storage
import io

# Rutas en Google Cloud Storage (GCS)
GCS_TRAIN_PATH = "gs://testoneml/latam-model/training/data-post-feature-eng/data_train.csv"
GCS_MODEL_PATH = "gs://testoneml/latam-model/inference/binary-model/xgb_model.pkl"

class DelayModel:
    def __init__(self):
        self._model = None

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0 / n_y1

        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        self._model.fit(features, target)

        # Guardar el modelo en memoria y subirlo a GCS
        self.upload_model_to_gcs()

    def upload_model_to_gcs(self):
        """Guarda el modelo en un bucket de Google Cloud Storage"""
        client = storage.Client()
        bucket_name, blob_path = self._parse_gcs_path(GCS_MODEL_PATH)
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Serializar modelo a bytes y subirlo a GCS
        model_bytes = io.BytesIO()
        pickle.dump(self._model, model_bytes)
        model_bytes.seek(0)

        blob.upload_from_file(model_bytes, content_type="application/octet-stream")
        print(f"✅ Modelo guardado en GCS: {GCS_MODEL_PATH}")

    def _parse_gcs_path(self, gcs_path):
        """Extrae el nombre del bucket y la ruta del blob desde una URL gs://"""
        path_parts = gcs_path.replace("gs://", "").split("/", 1)
        return path_parts[0], path_parts[1]

def load_data_from_gcs():
    """Carga el dataset desde un bucket de Google Cloud Storage"""
    client = storage.Client()
    bucket_name, blob_path = GCS_TRAIN_PATH.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)

    # Descargar el contenido como un DataFrame
    data_bytes = io.BytesIO(blob.download_as_bytes())
    return pd.read_csv(data_bytes)

if __name__ == "__main__":
    # Cargar datos desde GCS
    print(f"📥 Cargando dataset desde {GCS_TRAIN_PATH} ...")
    data = load_data_from_gcs()
    
    features = data.drop(columns=['delay'])
    target = data['delay']

    # División en conjunto de entrenamiento
    x_train, _, y_train, _ = train_test_split(features, target, test_size=0.33, random_state=42)
    
    # Entrenar y subir el modelo a GCS
    model = DelayModel()
    model.fit(x_train, y_train)
