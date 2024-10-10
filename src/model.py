import tensorflow as tf
from preprocessing import load_data, preprocess_data

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(1,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)  # Salida para temperatura
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train, epochs=10000, batch_size=32)
    return model

if __name__ == "__main__":
    data = load_data('data/temperature_data.csv')  # Cambia el nombre del archivo según sea necesario
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    model = create_model()
    trained_model = train_model(model, X_train, y_train)

    # Guardar el modelo entrenado
    trained_model.save('models/temp_predictor_model.h5')