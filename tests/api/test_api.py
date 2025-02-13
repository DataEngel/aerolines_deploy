import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from api_inference.api import app

class TestFlightDelayPredictionAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    @patch("api_inference.api.xgb_model")
    def test_should_get_predict(self, mock_model):
        """Prueba una predicción válida"""
        mock_model.predict.return_value = [0]
        mock_model.feature_names_in_ = ["OPERA_Grupo LATAM", "TIPOVUELO_N", "MES_7"]  # 🔹 Asegurar feature_names

        data = {"OPERA": "Grupo LATAM", "TIPOVUELO": "N", "MES": 7}
        response = self.client.post("/predict", json=data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"delay_prediction": 0})

    @patch("api_inference.api.xgb_model")
    def test_should_fail_invalid_mes(self, mock_model):
        """Falla cuando MES está fuera del rango esperado"""
        mock_model.predict.return_value = [0]
        mock_model.feature_names_in_ = ["OPERA_Grupo LATAM", "TIPOVUELO_N", "MES_7"]

        data = {"OPERA": "Grupo LATAM", "TIPOVUELO": "N", "MES": 13}  # ❌ Valor fuera de rango
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    @patch("api_inference.api.xgb_model")
    def test_should_fail_invalid_tipovuelo(self, mock_model):
        """Falla cuando TIPOVUELO tiene un valor no permitido"""
        mock_model.predict.return_value = [0]
        mock_model.feature_names_in_ = ["OPERA_Grupo LATAM", "TIPOVUELO_N", "MES_7"]

        data = {"OPERA": "Grupo LATAM", "TIPOVUELO": "X", "MES": 7}  # ❌ Valor no esperado
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

    @patch("api_inference.api.xgb_model")
    def test_should_fail_invalid_opera(self, mock_model):
        """Falla cuando OPERA tiene un valor desconocido"""
        mock_model.predict.return_value = [0]
        mock_model.feature_names_in_ = ["OPERA_Grupo LATAM", "TIPOVUELO_N", "MES_7"]

        data = {"OPERA": "Desconocida", "TIPOVUELO": "N", "MES": 7}  # ❌ Aerolínea desconocida
        response = self.client.post("/predict", json=data)
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
