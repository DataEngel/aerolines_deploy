import unittest
import pandas as pd
import sys
from pathlib import Path
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

# Agregar la ruta de model_script al PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent / "model_script"))

from model import DelayModel  # Importación corregida

class TestModel(unittest.TestCase):

    FEATURES_COLS = [
        "OPERA_Latin American Wings", 
        "MES_7",
        "MES_10",
        "OPERA_Grupo LATAM",
        "MES_12",
        "TIPOVUELO_I",
        "MES_4",
        "MES_11",
        "OPERA_Sky Airline",
        "OPERA_Copa Air"
    ]

    TARGET_COL = ["delay"]

    def setUp(self) -> None:
        """ Configuración inicial: carga los datos y crea una instancia del modelo """
        super().setUp()
        self.model = DelayModel()

        # Definir la ruta del dataset de pruebas
        DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.csv"

        # Manejo de errores si el dataset no existe
        if not DATA_PATH.exists():
            print(f"⚠️ Advertencia: El archivo {DATA_PATH} no existe. Se usará un DataFrame vacío.")
            self.data = pd.DataFrame(columns=self.FEATURES_COLS)
        else:
            self.data = pd.read_csv(DATA_PATH)

    def test_model_preprocess_for_training(self):
        """ Prueba que el preprocesamiento de entrenamiento genere las columnas correctas """
        features, target = self.model.preprocess(self.data, target_column="delay")

        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(features.shape[1], len(self.FEATURES_COLS))
        self.assertSetEqual(set(features.columns), set(self.FEATURES_COLS))

        self.assertIsInstance(target, pd.DataFrame)
        self.assertEqual(target.shape[1], len(self.TARGET_COL))
        self.assertSetEqual(set(target.columns), set(self.TARGET_COL))

    def test_model_preprocess_for_serving(self):
        """ Prueba que el preprocesamiento de inferencia genere las columnas correctas """
        features = self.model.preprocess(self.data)

        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(features.shape[1], len(self.FEATURES_COLS))
        self.assertSetEqual(set(features.columns), set(self.FEATURES_COLS))

    def test_model_fit(self):
        """ Prueba que el modelo se entrene correctamente y sus métricas sean válidas """
        features, target = self.model.preprocess(self.data, target_column="delay")
        _, features_validation, _, target_validation = train_test_split(features, target, test_size=0.33, random_state=42)

        self.model.fit(features=features, target=target)

        # Asegurar que el modelo se ha entrenado
        self.assertIsNotNone(self.model._model, "❌ El modelo no se entrenó correctamente.")

        predicted_target = self.model._model.predict(features_validation)
        report = classification_report(target_validation, predicted_target, output_dict=True)

        self.assertLessEqual(report["0"]["recall"], 0.60)
        self.assertLessEqual(report["0"]["f1-score"], 0.70)
        self.assertGreaterEqual(report["1"]["recall"], 0.60)
        self.assertGreaterEqual(report["1"]["f1-score"], 0.30)

    def test_model_predict(self):
        """ Prueba que el modelo genere predicciones válidas """
        features = self.model.preprocess(self.data)

        # Verificar que el DataFrame no esté vacío antes de predecir
        if features.empty:
            self.fail("❌ El DataFrame de features está vacío, no se puede predecir.")

        # Verificar que el modelo esté entrenado antes de predecir
        if self.model._model is None:
            self.fail("❌ El modelo no ha sido entrenado. No se puede realizar la predicción.")

        predicted_targets = self.model.predict(features=features)

        self.assertIsInstance(predicted_targets, list)
        self.assertEqual(len(predicted_targets), features.shape[0])
        self.assertTrue(all(isinstance(pred, int) for pred in predicted_targets))

if __name__ == "__main__":
    unittest.main()
