import pandas as pd
import pickle
import logging
import xgboost as xgb
from sklearn.model_selection import train_test_split
from pathlib import Path

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

from pathlib import Path

# Obtener la ruta absoluta del directorio raíz del proyecto
ROOT_DIR = Path(__file__).resolve().parent.parent  # Subir un nivel desde model_script

# Definir rutas absolutas a los archivos
TRAIN_DATA_PATH = ROOT_DIR / "data" / "data_train.csv"
INF_DATA_PATH = ROOT_DIR / "data" / "data_inf.csv"
MODEL_PATH = ROOT_DIR / "model_script" / "xgb_model.pkl"
OUTPUT_PRED_PATH = ROOT_DIR / "data" / "predictions.csv"

class DelayModel:
    def __init__(self):
        self._model = None

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        """Entrena un modelo XGBoost y lo guarda en un archivo."""
        logging.info("🚀 Iniciando entrenamiento del modelo...")
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0 / n_y1

        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        self._model.fit(features, target)

        with open(MODEL_PATH, "wb") as file:
            pickle.dump(self._model, file)
        logging.info(f"✅ Modelo guardado en {MODEL_PATH}")

    def load_model(self) -> None:
        """Carga el modelo desde un archivo pickle."""
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as file:
                self._model = pickle.load(file)
            logging.info(f"✅ Modelo cargado desde {MODEL_PATH}")
        else:
            logging.error(f"❌ Error: El archivo {MODEL_PATH} no existe.")
            raise FileNotFoundError(f"El modelo no se encontró en {MODEL_PATH}")

    def predict(self, features: pd.DataFrame) -> list:
        """Realiza predicciones con el modelo cargado."""
        if self._model is None:
            raise ValueError("❌ Error: El modelo no ha sido cargado.")
        return self._model.predict(features).tolist()


def load_data(file_path: Path) -> pd.DataFrame:
    """Carga datos desde un archivo CSV."""
    if file_path.exists():
        return pd.read_csv(file_path)
    else:
        logging.error(f"❌ Error: El archivo {file_path} no existe.")
        raise FileNotFoundError(f"Archivo no encontrado: {file_path}")


def main():
    model = DelayModel()
    
    # Cargar modelo si existe, si no, entrenar uno nuevo
    try:
        model.load_model()
    except FileNotFoundError:
        logging.info("⚠️ No se encontró un modelo, se procederá a entrenarlo.")
        data = load_data(TRAIN_DATA_PATH)
        features = data.drop(columns=['delay'])
        target = data['delay']
        x_train, _, y_train, _ = train_test_split(features, target, test_size=0.33, random_state=42)
        model.fit(x_train, y_train)
    
    # Cargar datos de inferencia y realizar predicciones
    data_inf = load_data(INF_DATA_PATH)
    predictions = model.predict(data_inf)
    data_inf["predicted_delay"] = predictions
    data_inf.to_csv(OUTPUT_PRED_PATH, index=False)
    logging.info(f"✅ Predicciones guardadas en {OUTPUT_PRED_PATH}")


if __name__ == "__main__":
    main()