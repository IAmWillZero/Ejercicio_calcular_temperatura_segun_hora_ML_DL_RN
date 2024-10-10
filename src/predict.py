import numpy as np
import tensorflow as tf

def load_trained_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_temperature(model, hour):
    if hour < 0 or hour > 23:
        raise ValueError("La hora debe estar en el rango de 0 a 23.")
    
    hour_scaled = np.array([[hour]])  # Convertir a array 2D
    prediction = model.predict(hour_scaled)
    return prediction[0][0]

if __name__ == "__main__":
    model = load_trained_model('models/temp_predictor_model.h5')
    
    # Ejemplo de predicción para la hora 15 (3 PM)
    hour_to_predict = 15
    predicted_temp = predict_temperature(model, hour_to_predict)
    
    print(f"Predicción de temperatura para las {hour_to_predict}: {predicted_temp:.2f}°C")