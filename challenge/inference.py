import pandas as pd
import pickle

MODEL_PATH = "../xgb_model.pkl"
INF_PATH = "../data/data_inf.csv"
OUTPUT_PRED_PATH = "../data/predictions.csv"

class DelayModel:
    def __init__(self):
        self._model = None

    def load_model(self) -> None:
        with open(MODEL_PATH, "rb") as file:
            self._model = pickle.load(file)
        print(f"✅ Modelo cargado desde {MODEL_PATH}")

    def predict(self, features: pd.DataFrame) -> list:
        if self._model is None:
            raise ValueError("El modelo no ha sido cargado.")
        return self._model.predict(features).tolist()

if __name__ == "__main__":
    model = DelayModel()
    model.load_model()

    data_inf = pd.read_csv(INF_PATH)
    predictions = model.predict(data_inf)

    data_inf['predicted_delay'] = predictions
    data_inf.to_csv(OUTPUT_PRED_PATH, index=False)
    print(f"✅ Predicciones guardadas en {OUTPUT_PRED_PATH}")
