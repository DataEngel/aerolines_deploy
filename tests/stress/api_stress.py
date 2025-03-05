from locust import HttpUser, task, between

class StressUser(HttpUser):
    wait_time = between(1, 5)  # Simula espera entre solicitudes

    @task
    def predict_argentinas(self):
        self.client.post(
            "/predict",
            headers={"Content-Type": "application/json"},
            json={
                "OPERA": "Aerolineas Argentinas",
                "MES": 3,
                "TIPOVUELO": "N"
            }
        )

    @task
    def predict_latam(self):
        self.client.post(
            "/predict",
            headers={"Content-Type": "application/json"},
            json={
                "OPERA": "Grupo LATAM",
                "MES": 7,
                "TIPOVUELO": "I"
            }
        )
