import pandas as pd
import pickle
import logging
from pathlib import Path

# Configuración de logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Definición de rutas
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "xgb_model.pkl"
INF_PATH = Path(__file__).resolve().parent.parent / "data" / "data_inf.csv"
OUTPUT_PRED_PATH = Path(__file__).resolve().parent.parent / "data" / "predictions.csv"

class DelayModel:
    def __init__(self):
        self._model = None

    def load_model(self, model_path: Path = None) -> None:
        """Carga el modelo desde un archivo pickle."""
        model_path = model_path or DEFAULT_MODEL_PATH

        try:
            with open(model_path, "rb") as file:
                self._model = pickle.load(file)
            logging.info(f"✅ Modelo cargado desde {model_path}")
        except FileNotFoundError:
            logging.error(f"❌ Error: El archivo {model_path} no existe.")
            raise
        except Exception as e:
            logging.error(f"❌ Error al cargar el modelo: {e}")
            raise

    def predict(self, features: pd.DataFrame) -> list:
        """Realiza predicciones asegurando que las columnas coincidan con el modelo."""
        if self._model is None:
            raise ValueError("❌ Error: El modelo no ha sido cargado.")

        expected_features = self._model.feature_names_in_

        # Validar que todas las columnas requeridas están en los datos de entrada
        missing_columns = [col for col in expected_features if col not in features.columns]
        extra_columns = [col for col in features.columns if col not in expected_features]

        if missing_columns or extra_columns:
            raise ValueError(f"❌ Error: Las columnas de entrada no coinciden con el modelo. "
                            f"Faltan: {missing_columns}, Sobran: {extra_columns}")

        # Asegurar el orden de las columnas
        features = features[expected_features]

        # Convertir todas las columnas a float para evitar errores con XGBoost
        features = features.astype(float)

        try:
            predictions = self._model.predict(features).tolist()
            return predictions
        except Exception as e:
            logging.error(f"❌ Error en la predicción: {e}")
            raise


def main():
    """Carga el modelo, realiza predicciones y guarda los resultados."""
    model = DelayModel()

    # Cargar el modelo
    model.load_model()

    # Cargar los datos de inferencia
    if not INF_PATH.exists():
        logging.error(f"❌ Error: El archivo {INF_PATH} no existe.")
        return

    data_inf = pd.read_csv(INF_PATH)

    # Realizar predicciones
    predictions = model.predict(data_inf)

    # Guardar las predicciones en un nuevo archivo
    data_inf["predicted_delay"] = predictions
    data_inf.to_csv(OUTPUT_PRED_PATH, index=False)
    logging.info(f"✅ Predicciones guardadas en {OUTPUT_PRED_PATH}")

if __name__ == "__main__":
    main()
