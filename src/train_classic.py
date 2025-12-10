
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib
from src.preprocessing import limpiar_texto 

def entrenar_modelo_clasico(X_train_raw, y_train, model_path='models/modelo_svm.pkl'):
    """Crea un pipeline TF-IDF + SVM, lo entrena y lo guarda."""

    modelo_pipeline = Pipeline([

        ('tfidf', TfidfVectorizer(preprocessor=limpiar_texto, max_features=5000)),
        ('svm', SVC(probability=True, random_state=42))
    ])
    
    # 2. Entrenamiento
    print("Iniciando entrenamiento del modelo clásico (SVM)...")
    modelo_pipeline.fit(X_train_raw, y_train)
    print("Entrenamiento de SVM completado.")

    # 3. Guardar el modelo entrenado
    joblib.dump(modelo_pipeline, model_path)
    print(f"Modelo SVM guardado en: {model_path}")
    
    return modelo_pipeline