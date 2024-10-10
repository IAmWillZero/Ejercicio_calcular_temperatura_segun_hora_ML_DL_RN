import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Filtrar solo las columnas necesarias
    X = data[['hour']].values
    y = data['temperature'].values

    # Validar que las horas estén en el rango de 0 a 23
    if not all(0 <= hour <= 23 for hour in X.flatten()):
        raise ValueError("Las horas deben estar en el rango de 0 a 23.")

    # Normalización de los datos
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, scaler