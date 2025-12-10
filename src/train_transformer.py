from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf
import numpy as np

MODEL_NAME = 'dccuchile/bert-base-spanish-wwm-cased' 
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3

def codificar_datos(textos, etiquetas, tokenizer):
    """Convierte texto en formato de entrada (Input IDs, Attention Masks) para BERT."""
    input_ids = []
    attention_masks = []
    
    for texto in textos:
        encoded = tokenizer.encode_plus(
            texto,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        attention_masks.append(encoded['attention_mask'])
        
    return [np.array(input_ids), np.array(attention_masks)], np.array(etiquetas)

def entrenar_modelo_transformer(X_train_raw, X_test_raw, y_train, y_test, model_path='models/bert_fakenews'):
    """Carga BETO, realiza el fine-tuning y guarda el modelo."""
    

    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    X_train_data, y_train_labels = codificar_datos(X_train_raw.tolist(), y_train.tolist(), tokenizer)
    X_test_data, y_test_labels = codificar_datos(X_test_raw.tolist(), y_test.tolist(), tokenizer)
    

    model = TFBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    

    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    print("\n# Iniciando Fine-tuning del Modelo Transformer (BETO) ##")
    model.fit(
        X_train_data, 
        y_train_labels, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        validation_data=(X_test_data, y_test_labels)
    )
    

    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path) # Es vital guardar el tokenizer también
    print(f"Modelo BERT guardado en: {model_path}")
    
    return model, X_test_data, y_test_labels