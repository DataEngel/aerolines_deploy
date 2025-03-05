# Usa una imagen oficial de Python como base
FROM python:3.10.12


# Establece el directorio de trabajo en el contenedor
WORKDIR /app

# Instala dependencias del sistema necesarias
RUN apt-get update -y && apt-get install -y gcc

# Copia los archivos de requisitos y los instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Crea el directorio models dentro del contenedor
RUN mkdir -p /app/models

# Copia el modelo entrenado dentro del contenedor
COPY xgb_model.pkl /app/models/xgb_model.pkl

# Copia el resto del código de la aplicación
COPY . .

# Expone el puerto en el que correrá la API
EXPOSE 8080


# Comando para ejecutar la aplicación con Uvicorn
#CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
CMD ["python", "-m", "uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]