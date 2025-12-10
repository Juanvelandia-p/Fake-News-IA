import os
import tensorflow as tf
from src.preprocessing import cargar_datos_mixtos 
from src.train_classic import entrenar_modelo_clasico
from src.train_transformer import entrenar_modelo_transformer
from src.evaluation import evaluar_modelo_clasico, evaluar_modelo_transformer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.get_logger().setLevel('ERROR')



def ejecutar_flujo_completo():
    """Ejecuta la carga de datos, entrenamiento y evaluación de ambos modelos."""
    
    print("--- FASE 1: Preparación de Datos ---")
    
    # 1. Carga y Unificación de Datos
    X_train_raw, X_test_raw, y_train, y_test = cargar_datos_mixtos(
        ruta_excel_train='data/train.xlsx', 
        ruta_csv_train='data/esp_fake_news.csv', # Nueva ruta CSV para entrenamiento
        ruta_excel_test='data/test.xlsx'
    )
    
    print(f"Datos de Entrenamiento: {len(X_train_raw)} | Datos de Prueba: {len(X_test_raw)}")

    # 2. Entrenamiento y Evaluación del Modelo Clásico (SVM)
    # Las funciones de entrenamiento y evaluación NO necesitan cambiarse
    # ya que ya están diseñadas para recibir los conjuntos separados.
    print("--- FASE 2: Entrenamiento y Evaluación de Baseline (SVM) ---")
    modelo_svm = entrenar_modelo_clasico(X_train_raw, y_train)
    evaluar_modelo_clasico(modelo_svm, X_test_raw, y_test)
    
    print("\n" + "="*60 + "\n")

    # 3. Entrenamiento y Evaluación del Modelo Transformer (BETO)
    print("--- FASE 3: Entrenamiento y Evaluación de Transformer (BETO) ---")
    modelo_bert, X_test_data_bert, y_test_labels_bert = entrenar_modelo_transformer(
        X_train_raw, X_test_raw, y_train, y_test
    )
    evaluar_modelo_transformer(modelo_bert, X_test_data_bert, y_test_labels_bert)

if __name__ == "__main__":
    if not os.path.exists('models'):
        os.makedirs('models')
    ejecutar_flujo_completo()