import unittest
import pandas as pd
from pathlib import Path
from challenge.model_inf import DelayModel

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        """Configura el entorno antes de cada prueba."""
        super().setUp()
        self.model = DelayModel()
        
        # Ajustar la ruta correcta del modelo
        model_path = Path(__file__).resolve().parents[2] / "challenge" / "xgb_model.pkl"

        # Validar existencia del archivo antes de continuar
        if not model_path.exists():
            self.fail(f"El archivo del modelo no se encontró en {model_path}. Asegúrate de que el modelo ha sido entrenado y guardado correctamente.")

        # Cargar modelo
        self.model.load_model(model_path)

        # Obtener las columnas esperadas del modelo entrenado
        self.expected_features = self.model._model.feature_names_in_

        # Crear un DataFrame de prueba con las columnas esperadas (convertidas a float)
        self.data = pd.DataFrame({col: [0.0, 1.0] for col in self.expected_features}, dtype=float)

    def test_model_predict(self):
        """Prueba que el modelo genera predicciones correctamente."""
        predicted_targets = self.model.predict(features=self.data)

        self.assertIsInstance(predicted_targets, list)
        self.assertEqual(len(predicted_targets), self.data.shape[0])
        self.assertTrue(all(isinstance(pred, (int, float)) for pred in predicted_targets))

    def test_model_empty_input(self):
        """Prueba el comportamiento del modelo con un DataFrame vacío."""
        empty_df = pd.DataFrame(columns=self.expected_features, dtype=float)
        predicted_targets = self.model.predict(features=empty_df)

        self.assertEqual(predicted_targets, [])  # Debe devolver una lista vacía

    def test_model_invalid_input(self):
        """Prueba el comportamiento del modelo con columnas incorrectas."""
        invalid_data = pd.DataFrame({
            "INVALID_COLUMN": [1.0, 2.0, 3.0],
            "MES_7": [1.0, 0.0, 1.0]
        }, dtype=float)
        
        with self.assertRaises(ValueError):
            self.model.predict(features=invalid_data)

if __name__ == "__main__":
    unittest.main()
