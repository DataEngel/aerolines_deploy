import logging
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from pydantic import BaseModel

# Configurar logging para FastAPI
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Intentar cargar el modelo
model_path = "xgb_model.pkl"
try:
    xgb_model = joblib.load(model_path)
    logger.info("✅ Modelo cargado exitosamente")
    logger.info(f"🔍 Features del modelo: {xgb_model.feature_names_in_}")
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

# Endpoint para hacer predicciones
@app.post("/predict")
def predict_delay(flight_data: FlightData):
    """
    Recibe datos de un vuelo, los preprocesa y devuelve la predicción de retraso.

    Args:
        flight_data (FlightData): Datos del vuelo en formato JSON.

    Returns:
        dict: Predicción de retraso (1: retraso, 0: no retraso).
    """
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

        # Seleccionar solo las características utilizadas en el modelo
        feature_cols = [
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
        
        # Asegurar que todas las columnas necesarias estén presentes
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0

        # Verificar si faltan columnas
        missing_features = [col for col in feature_cols if col not in df.columns]
        if missing_features:
            logger.error(f"❌ Faltan columnas en la entrada: {missing_features}")
            raise HTTPException(status_code=400, detail=f"Columnas faltantes: {missing_features}")

        # Hacer la predicción
        prediction = xgb_model.predict(df[feature_cols])
        if len(prediction) == 0:
            logger.error("❌ El modelo devolvió una predicción vacía")
            raise HTTPException(status_code=500, detail="Error en la predicción")

        logger.info(f"✅ Predicción generada: {prediction[0]}")
        return {"delay_prediction": int(prediction[0])}

    except Exception as e:
        logger.error(f"❌ Error en el proceso de predicción: {e}")
        raise HTTPException(status_code=500, detail="Error en el procesamiento de la predicción")
