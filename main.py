import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocessing import cargar_datos_mixtos, limpiar_texto
from src.train_transformer import entrenar_modelo_transformer
from src.evaluation import evaluar_modelo_clasico, evaluar_modelo_transformer


def ejecutar_flujo_completo():
    """
    Ejecuta el flujo completo: carga, preprocesamiento, cálculo de pesos, 
    entrenamiento de modelos (SVM y BETO) y evaluación.
    """
    
    # 1. Carga y preprocesamiento de datos (incluye balanceo 1:1)
    X_train_raw, X_test_raw, y_train, y_test = cargar_datos_mixtos(
        ruta_excel_train='data/train.xlsx', 
        ruta_csv_train='data/esp_fake_news.csv', 
        ruta_excel_test='data/test.xlsx'
    )

    # 2. CÁLCULO DE PESOS DE CLASE para corregir el sesgo en BERT
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
    class_weights_dict = dict(zip(classes, weights))
    
    print("-" * 50)
    print(f"Pesos de Clase Calculados (0: Falsa, 1: Real): {class_weights_dict}")
    print("-" * 50)


    # --- CLASIFICADOR CLÁSICO (LÍNEA BASE) ---

    # 3. Preprocesamiento para SVM (Limpieza de texto y Vectorización)
    print("\n-> Entrenando Clasificador Clásico (SVM) como baseline...")
    X_train_clean = X_train_raw.apply(limpiar_texto)
    X_test_clean = X_test_raw.apply(limpiar_texto)
    
    # 4. Definición y Entrenamiento de Pipeline (TF-IDF + SVM)
    pipeline_svm = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)),
        ('clf', LinearSVC(class_weight='balanced', random_state=42, dual=True)) # Balanceado para el SVM
    ])
    
    pipeline_svm.fit(X_train_clean, y_train)

    # 5. Evaluación del Clasificador Clásico
    evaluar_modelo_clasico(pipeline_svm, X_test_clean, y_test)


    # --- MODELO TRANSFORMER (SOLUCIÓN AVANZADA) ---

    # 6. Entrenamiento del Modelo Transformer (Pasa los pesos de clase)
    model_bert, X_test_data_bert, y_test_labels_bert = entrenar_modelo_transformer(
        X_train_raw, y_train, X_test_raw, y_test, class_weights_dict 
    )

    # 7. Evaluación del Modelo Transformer (Usa optimización de umbral)
    evaluar_modelo_transformer(model_bert, X_test_data_bert, y_test_labels_bert)

if __name__ == "__main__":
    ejecutar_flujo_completo()