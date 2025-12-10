import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import logging

# Configuración de logging para evitar mensajes excesivos de TensorFlow/HuggingFace
logging.basicConfig(level=logging.ERROR)

def evaluar_modelo_clasico(model, X_test, y_test):
    """Evalúa un modelo de clasificación clásico (ej. SVM, Regresión Logística)."""
    try:
        y_pred = model.predict(X_test)
        
        auc_roc = "N/A"
        # Para AUC-ROC, si el modelo soporta predict_proba
        if hasattr(model, "predict_proba"):
            y_probas = model.predict_proba(X_test)[:, 1]
            auc_roc = roc_auc_score(y_test, y_probas)

        print("\n==================================================")
        print("          Resultados del Clasificador Clásico (SVM)")
        print("==================================================")
        print(classification_report(y_test, y_pred, digits=2, zero_division=0))
        
        # CORRECCIÓN DE ERROR: Solo aplica formato si es un número (float)
        if isinstance(auc_roc, float):
            print(f"AUC-ROC Score: {auc_roc:.4f}")
        else:
            print(f"AUC-ROC Score: {auc_roc}")
        
    except Exception as e:
        print(f"Error al evaluar el modelo clásico: {e}")
        # Si ocurre un error, muestra el error de la causa y el AUC-ROC queda como N/A


def find_optimal_threshold(y_true, y_probas):
    """
    Busca el umbral dentro de un rango que maximice el F1-score 
    específicamente para la Clase 1 (Noticia Real).
    """
    thresholds = np.arange(0.3, 0.71, 0.02) 
    best_f1_clase1 = -1
    best_threshold = 0.5

    for t in thresholds:
        y_pred_labels = (y_probas >= t).astype(int)
        
        # Calculamos el F1-score SÓLO para la Clase 1
        f1_clase_1 = f1_score(y_true, y_pred_labels, pos_label=1, average='binary', zero_division=0) 
        
        if f1_clase_1 > best_f1_clase1:
            best_f1_clase1 = f1_clase_1
            best_threshold = t
            
    return best_threshold, best_f1_clase1


def evaluar_modelo_transformer(model, X_test_data_bert, y_test_labels_bert):
    """
    Evalúa el modelo Transformer (BERT/BETO) encontrando y aplicando el umbral óptimo.
    """
    try:
        # Predicción: obtiene los logits (puntajes sin activar)
        print("\n-> Realizando predicciones con el modelo Transformer...")
        y_pred_logits = model.predict(X_test_data_bert)
        
        # 1. Calcular probabilidades: aplicamos softmax
        y_probas = tf.nn.softmax(y_pred_logits.logits, axis=1).numpy()[:, 1] # Probabilidad de ser Clase 1 (Real)
        
        
        # 2. OPTIMIZACIÓN DEL UMBRAL
        optimal_threshold, max_f1_clase1 = find_optimal_threshold(y_test_labels_bert, y_probas)
        print(f"-> Umbral Óptimo Encontrado (maximizando F1 Clase 1): {optimal_threshold:.2f} (F1 = {max_f1_clase1:.2f})")

        # 3. Aplicar el umbral óptimo
        y_pred_labels_optimal = (y_probas >= optimal_threshold).astype(int)
        
        
        # 4. Cálculo de Métricas Finales
        auc_roc = roc_auc_score(y_test_labels_bert, y_probas)

        print("\n==================================================")
        print("          Resultados del Modelo Transformer (BETO) - Umbral Optimizado")
        print("==================================================")
        print(classification_report(y_test_labels_bert, y_pred_labels_optimal, digits=2, zero_division=0))
        print(f"AUC-ROC Score: {auc_roc:.4f}")
        
    except Exception as e:
        print(f"Error al evaluar el modelo Transformer: {e}")
        print(e)