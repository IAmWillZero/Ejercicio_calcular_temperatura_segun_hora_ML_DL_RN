import flet as ft
import numpy as np

def visualize_results():
    def main(page: ft.Page):
        page.title = "Predicción de Temperatura"
        
        hours = np.arange(0, 24)  # Horas del día (0 a 23)
        
        from predict import load_trained_model, predict_temperature
        
        model = load_trained_model('models/temp_predictor_model.h5')
        
        predictions = []
        for hour in hours:
            try:
                prediction = predict_temperature(model, hour)
                predictions.append(prediction)
                page.add(ft.Text(f"Hora: {hour} - Predicción: {prediction:.2f}°C"))
            except ValueError as e:
                page.add(ft.Text(f"Error: {e}"))
    
    ft.app(target=main)

if __name__ == "__main__":
    visualize_results()