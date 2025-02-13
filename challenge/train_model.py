import pandas as pd
import pickle 
import xgboost as xgb
from sklearn.model_selection import train_test_split

TRAIN_PATH = "../data/data_train.csv"
MODEL_PATH = "xgb_model.pkl"

class DelayModel:
    def __init__(self):
        self._model = None

    def fit(self, features: pd.DataFrame, target: pd.DataFrame) -> None:
        n_y0 = len(target[target == 0])
        n_y1 = len(target[target == 1])
        scale = n_y0 / n_y1

        self._model = xgb.XGBClassifier(random_state=1, learning_rate=0.01, scale_pos_weight=scale)
        self._model.fit(features, target)

        with open(MODEL_PATH, "wb") as file:
            pickle.dump(self._model, file)
        print(f"✅ Modelo guardado en {MODEL_PATH}")

if __name__ == "__main__":
    data = pd.read_csv(TRAIN_PATH)
    features = data.drop(columns=['delay'])
    target = data['delay']

    x_train, _, y_train, _ = train_test_split(features, target, test_size=0.33, random_state=42)
    
    model = DelayModel()
    model.fit(x_train, y_train)
