# TempPredictor

## Descripción
TempPredictor es un proyecto que utiliza redes neuronales para predecir la temperatura en función de la hora del día. Este modelo permite realizar predicciones precisas basadas en datos históricos.

## Requisitos
- Python 3
- Las siguientes bibliotecas deben estar instaladas:
  - numpy
  - pandas
  - scikit-learn
  - tensorflow
  - matplotlib
  - flet

## Lógica de Solución
1. **Preprocesamiento**: Los datos se limpian y transforman para ser utilizados en el modelo.
2. **Modelo**: Se implementa una red neuronal utilizando TensorFlow/Keras con capas densas.
3. **Entrenamiento**: El modelo se entrena con un conjunto de datos dividido en entrenamiento y prueba.
4. **Evaluación**: Se evalúa el rendimiento del modelo utilizando métricas como RMSE (Root Mean Squared Error).
5. **Predicción**: Se realizan predicciones en horas no vistas utilizando el modelo entrenado.
6. **Visualización**: Se presentan los resultados utilizando Flet.

## Ejecución
Para ejecutar el proyecto, siga estos pasos:
1. Instale las dependencias desde `requirements.txt`:
   ```bash
   pip install -r requirements.txt

2. Entrenar el modelo:
    Ejecuta el script model.py:
    ```bash
    python src/model.py

3. Realizar predicciones:
    Ejecuta el script predict.py:
    ```bash
    python src/predict.py

4. Visualizar resultados:
    Ejecuta el script visualization.py:
    ```bash
    python src/visualization.py