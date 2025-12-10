import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score
import logging

# Configuración de logging para evitar mensajes excesivos de TensorFlow/HuggingFace
logging.basicConfig(level=logging.ERROR)

def evaluar_modelo_clasico(model, X_test, y_test):
    """Evalúa un modelo de clasificación clásico (ej. SVM, Regresión Logística)."""
    try:
        y_pred = model.predict(X_test)
        
        # Para AUC-ROC, si el modelo soporta predict_proba
        if hasattr(model, "predict_proba"):
            y_probas = model.predict_proba(X_test)[:, 1]
            auc_roc = roc_auc_score(y_test, y_probas)
        else:
            auc_roc = "N/A" # No siempre está disponible en modelos básicos

        print("\n==================================================")
        print("          Resultados del Clasificador Clásico (SVM)")
        print("==================================================")
        print(classification_report(y_test, y_pred, digits=2, zero_division=0))
        print(f"AUC-ROC Score: {auc_roc:.4f}")
        
    except Exception as e:
        print(f"Error al evaluar el modelo clásico: {e}")


def evaluar_modelo_transformer(model, X_test_data_bert, y_test_labels_bert):
    """
    Evalúa el modelo Transformer (BERT/BETO) e implementa un ajuste de umbral 
    para optimizar el Recall de la clase minoritaria (Clase 1 - Real).
    """
    try:
        # Predicción: obtiene los logits (puntajes sin activar)
        print("\n-> Realizando predicciones con el modelo Transformer...")
        y_pred_logits = model.predict(X_test_data_bert)
        
        # 1. Calcular probabilidades: aplicamos softmax para convertir logits a probabilidades
        # Nos interesa la columna [:, 1] que es la probabilidad de ser Clase 1 (Real)
        y_probas = tf.nn.softmax(y_pred_logits.logits, axis=1).numpy()[:, 1]
        
        
        # 2. IMPLEMENTACIÓN Y AJUSTE DE UMBRAL (THRESHOLD)
        # Objetivo: Aumentar el Recall de la Clase 1. Bajamos el umbral a 0.40 (o menos).
        UMBRAL_OPTIMO = 0.40  # ¡PRUEBA CON ESTE VALOR!
        print(f"-> Usando Umbral de Decisión (Clase 1) de: {UMBRAL_OPTIMO}")

        # Convertir probabilidades a etiquetas binarias usando el nuevo umbral
        y_pred_labels = (y_probas >= UMBRAL_OPTIMO).astype(int)
        
        
        # 3. Cálculo de Métricas Finales
        auc_roc = roc_auc_score(y_test_labels_bert, y_probas)

        print("\n==================================================")
        print("          Resultados del Modelo Transformer (BETO)")
        print("==================================================")
        print(classification_report(y_test_labels_bert, y_pred_labels, digits=2, zero_division=0))
        print(f"AUC-ROC Score: {auc_roc:.4f}")
        
    except Exception as e:
        print(f"Error al evaluar el modelo Transformer: {e}")
        print(e)
        

# Nota: Debes asegurarte de que tu main.py llame a las funciones
# de evaluación con los respectivos datos preprocesados y modelos entrenados.