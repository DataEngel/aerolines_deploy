import pandas as pd
from datetime import datetime

DATA_PATH = "../data/data.csv"
OUTPUT_TRAIN_PATH = "../data/data_train.csv"

def get_period_day(date: str) -> str:
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    if datetime.strptime("05:00", '%H:%M').time() <= date_time <= datetime.strptime("11:59", '%H:%M').time():
        return 'mañana'
    elif datetime.strptime("12:00", '%H:%M').time() <= date_time <= datetime.strptime("18:59", '%H:%M').time():
        return 'tarde'
    else:
        return 'noche'

def is_high_season(fecha: str) -> int:
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    ranges = [
        (datetime.strptime('15-Dec', '%d-%b'), datetime.strptime('31-Dec', '%d-%b')),
        (datetime.strptime('1-Jan', '%d-%b'), datetime.strptime('3-Mar', '%d-%b')),
        (datetime.strptime('15-Jul', '%d-%b'), datetime.strptime('31-Jul', '%d-%b')),
        (datetime.strptime('11-Sep', '%d-%b'), datetime.strptime('30-Sep', '%d-%b'))
    ]
    for r_min, r_max in ranges:
        if r_min.replace(year=fecha_año) <= fecha <= r_max.replace(year=fecha_año):
            return 1
    return 0

def get_min_diff(row: pd.Series) -> float:
    fecha_o = datetime.strptime(row['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(row['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    return (fecha_o - fecha_i).total_seconds() / 60

def preprocess_train(data: pd.DataFrame) -> pd.DataFrame:
    data['period_day'] = data['Fecha-I'].apply(get_period_day)
    data['high_season'] = data['Fecha-I'].apply(is_high_season)
    data['min_diff'] = data.apply(get_min_diff, axis=1)
    data['delay'] = (data['min_diff'] > 15).astype(int)

    features = pd.concat([
        pd.get_dummies(data['OPERA'], prefix='OPERA'),
        pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'), 
        pd.get_dummies(data['MES'], prefix='MES')
    ], axis=1)

    features['delay'] = data['delay']

    return features

if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    data_train = data.sample(frac=0.8, random_state=42).reset_index(drop=True)
    processed_train = preprocess_train(data_train)
    processed_train.to_csv(OUTPUT_TRAIN_PATH, index=False)
    print(f"✅ Archivo guardado en {OUTPUT_TRAIN_PATH}")
