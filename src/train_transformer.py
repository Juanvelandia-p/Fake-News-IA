import tensorflow as tf
from transformers import TFBertForSequenceClassification, BertTokenizer
from sklearn.model_selection import train_test_split
import numpy as np

# --- CONFIGURACIÓN DE HIPERPARÁMETROS ---
MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased' # Puedes usar BETO
MAX_LEN = 128
BATCH_SIZE = 32
# Mantener bajo el LR y las Épocas para evitar el sobreajuste (overfitting)
LEARNING_RATE = 2e-5 
EPOCHS = 5 

# Cargar el tokenizador y el modelo BERT pre-entrenado
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)


def encode_data(tokenizer, texts, max_len):
    """Tokeniza y codifica los textos para el modelo BERT."""
    return tokenizer.batch_encode_plus(
        texts.tolist(),
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        return_tensors='tf'
    )

def entrenar_modelo_transformer(X_train, y_train, X_test, y_test, class_weights_dict):
    """
    Prepara los datos, entrena el modelo BETO con Pesos de Clase, 
    y retorna el modelo entrenado y los datos de prueba.
    """
    
    # 1. Separar datos de entrenamiento en Entrenamiento y Validación (80/20)
    X_train_data, X_val, y_train_data, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # 2. Codificación (Tokenización) de los datasets
    print("-> Tokenizando datos de entrenamiento...")
    train_encodings = encode_data(tokenizer, X_train_data, MAX_LEN)
    val_encodings = encode_data(tokenizer, X_val, MAX_LEN)
    test_encodings = encode_data(tokenizer, X_test, MAX_LEN)
    
    # 3. Creación de Datasets de TensorFlow
    X_train_data_bert = tf.data.Dataset.from_tensor_slices((
        {'input_ids': train_encodings['input_ids'], 
         'attention_mask': train_encodings['attention_mask'], 
         'token_type_ids': train_encodings['token_type_ids']},
        y_train_data.values
    )).batch(BATCH_SIZE)

    X_val_data = tf.data.Dataset.from_tensor_slices((
        {'input_ids': val_encodings['input_ids'], 
         'attention_mask': val_encodings['attention_mask'], 
         'token_type_ids': val_encodings['token_type_ids']},
        y_val.values
    )).batch(BATCH_SIZE)
    
    X_test_data_bert = tf.data.Dataset.from_tensor_slices((
        {'input_ids': test_encodings['input_ids'], 
         'attention_mask': test_encodings['attention_mask'], 
         'token_type_ids': test_encodings['token_type_ids']},
    )).batch(BATCH_SIZE)
    
    y_test_labels_bert = y_test.values

    # 4. Inicialización y Compilación del Modelo
    model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    # Usamos from_logits=True porque TFBertForSequenceClassification ya tiene la capa de salida (logits)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]
    
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    # 5. Entrenamiento del Modelo con PESOS DE CLASE
    print("\n-> Entrenando modelo (BETO) con Pesos de Clase...")
    history = model.fit(
        X_train_data_bert, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_data=X_val_data,
        class_weight=class_weights_dict # <--- ¡Implementación de Pesos de Clase!
    )
    
    # 6. Retorno de Modelo y Datos de Prueba
    return model, X_test_data_bert, y_test_labels_bert