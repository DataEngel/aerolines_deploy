import logging
import os
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel
import uvicorn

# Configurar logging para FastAPI
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Cargar el modelo desde la ruta local en Docker
model_path = "/app/models/xgb_model.pkl"
try:
    xgb_model = joblib.load(model_path)
    logger.info(f"✅ Modelo cargado exitosamente desde {model_path}")
    feature_names = xgb_model.feature_names_in_  # Lista de las features esperadas por el modelo
    logger.info(f"🔍 Features del modelo: {feature_names}")
except Exception as e:
    logger.error(f"❌ Error al cargar el modelo: {e}")
    xgb_model = None

# Inicializar FastAPI
app = FastAPI(title="Flight Delay Prediction API")

# Definir la estructura de los datos de entrada
class FlightData(BaseModel):
    OPERA: str
    MES: int
    TIPOVUELO: str

@app.get("/health")
def health_check():
    return {"status": "ok"}

# Endpoint para hacer predicciones
@app.post("/predict")
def predict_delay(flight_data: FlightData):
    if xgb_model is None:
        logger.error("❌ Modelo no cargado. Verifica que xgb_model.pkl está disponible")
        raise HTTPException(status_code=500, detail="Modelo no disponible")

    # Registrar los datos recibidos
    logger.debug(f"🔍 JSON recibido: {flight_data.dict()}")

    try:
        # Convertir datos de entrada a DataFrame
        df = pd.DataFrame([flight_data.dict()])

        # Convertir variables categóricas en dummy variables
        df = pd.get_dummies(df, columns=["OPERA", "TIPOVUELO", "MES"], prefix=["OPERA", "TIPOVUELO", "MES"])
        logger.debug(f"📊 DataFrame después de get_dummies():\n{df}")

        # 🔹 Rellenar automáticamente columnas faltantes con 0
        for col in feature_names:
            if col not in df.columns:
                df[col] = 0

        # Ordenar las columnas en el mismo orden que el modelo
        df = df[list(feature_names)]

        # Confirmar la forma del DataFrame antes de la predicción
        logger.debug(f"📊 DataFrame final antes de predicción:\n{df.head()}")
        logger.debug(f"✅ Shape esperado: {len(feature_names)}, Shape actual: {df.shape}")

        # Hacer la predicción
        prediction = xgb_model.predict(df)
        if len(prediction) == 0:
            logger.error("❌ El modelo devolvió una predicción vacía")
            raise HTTPException(status_code=500, detail="Error en la predicción")

        logger.info(f"✅ Predicción generada: {prediction[0]}")
        return {"delay_prediction": int(prediction[0])}

    except Exception as e:
        logger.error(f"❌ Error en el proceso de predicción: {e}")
        raise HTTPException(status_code=500, detail="Error en el procesamiento de la predicción")

# Punto de entrada principal para ejecutar Uvicorn
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))  # Obtener el puerto desde la variable de entorno
    uvicorn.run(app, host="0.0.0.0", port=port)
