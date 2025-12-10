import pandas as pd
import spacy
import re
import os
from sklearn.model_selection import train_test_split # Se mantiene por si se usa en otro lugar, aunque ya no es necesaria para el split

# --- CONFIGURACIÓN INICIAL ---
try:
    # Carga el modelo pequeño de spaCy para español
    nlp = spacy.load("es_core_news_sm")
    stop_words = nlp.Defaults.stop_words
except OSError:
    print("Descargando modelo de spaCy...")
    spacy.cli.download("es_core_news_sm")
    nlp = spacy.load("es_core_news_sm")
    stop_words = nlp.Defaults.stop_words

def limpiar_texto(texto):
    """Aplica normalización, supresión de ruido, tokenización y lematización."""
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'http\S+|www\S+|https\S+', '', texto, flags=re.MULTILINE)
    texto = re.sub(r'[^\w\s]', '', texto) 
    doc = nlp(texto)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_space]
    return " ".join(tokens)


def cargar_csv_adicional(ruta_csv, etiqueta_fija):
    """Carga CSV con la columna 'text' y asigna una etiqueta fija (1 o 0)."""
    df = pd.read_csv(ruta_csv)
    df.columns = df.columns.str.lower()
    
    # Aseguramos que la columna 'text' exista en los nuevos CSVs
    if 'text' not in df.columns:
        # Algunos datasets usan 'headlines' o 'title' como columna de texto principal
        raise KeyError(f"El archivo {ruta_csv} no tiene una columna 'text' legible. Por favor, revisa el nombre de la columna.")
        
    df_final = df[['text']].copy()
    # Usamos las etiquetas categóricas 'True'/'Fake' para unificación
    df_final['label_cat'] = etiqueta_fija 
    return df_final


def cargar_datos_mixtos(ruta_excel_train, ruta_csv_train, ruta_excel_test):
    """
    Carga, balancea y unifica los datos de entrenamiento (Originales + Nuevos CSVs) 
    y carga el conjunto de prueba.
    """
    
    # --- 1. CARGA DE DATOS ORIGINALES DE ENTRENAMIENTO ---
    
    # 1a. Cargar Excel de entrenamiento (Train.xlsx - balance desequilibrado)
    df_excel_train = pd.read_excel(ruta_excel_train, sheet_name=0) 
    df_excel_train.columns = df_excel_train.columns.str.lower() 
    df_excel_train = df_excel_train.rename(columns={'text': 'text', 'category': 'label_cat'}) 
    
    # 1b. Cargar CSV de entrenamiento (esp_fake_news.csv - todas Falsas)
    df_csv_train = pd.read_csv(ruta_csv_train) 
    df_csv_train.columns = df_csv_train.columns.str.lower() 
    df_csv_train = df_csv_train.rename(columns={'fake statement': 'text'})
    df_csv_train['label_cat'] = 'Fake'
    
    # Unificación inicial del conjunto original
    df_train_original = pd.concat([df_excel_train[['text', 'label_cat']], df_csv_train[['text', 'label_cat']]], ignore_index=True)

    # Limpieza de etiquetas categóricas originales antes de balanceo
    df_train_original['label_cat'] = df_train_original['label_cat'].astype(str).str.strip() 
    
    # 2. --- BALANCEO DE CLASES Y ADICIÓN DE NUEVOS DATOS ---
    
    # 2a. Carga y Etiquetado de Nuevos Datos
    print("-> Cargando y balanceando datasets adicionales (1000 True / 1000 Fake)...")

    # RUTAS CORREGIDAS USANDO os.path.join
    RUTA_NEW_TRUE = os.path.join('data','archive', 'onlytrue1000.csv') 
    RUTA_NEW_FAKE = os.path.join('data', 'archive', 'onlyfakes1000.csv')

    df_new_true = cargar_csv_adicional(RUTA_NEW_TRUE, 'True')
    df_new_fake = cargar_csv_adicional(RUTA_NEW_FAKE, 'Fake')


    # 2b. Preparación y Submuestreo (¡ESTA ES LA SECCIÓN CLAVE A CAMBIAR!)

    # Unificamos todas las noticias Falsas disponibles (~2887 originales + 1000 nuevas)
    df_all_fake = pd.concat([
        df_train_original[df_train_original['label_cat'] == 'Fake'],
        df_new_fake
    ], ignore_index=True)

    df_original_real = df_train_original[df_train_original['label_cat'] == 'True'] 
    df_new_true = df_new_true # Las mantenemos todas


    # ¡NUEVO SUBMUESTREO AL 30% DE FALSAS!
    # Target: 573 noticias Falsas
    df_fake_sampled_final = df_all_fake.sample(n=1000, random_state=42)


    # 2c. Unificación Final del Conjunto de Entrenamiento (70% Real / 30% Falsa)
    df_train_final = pd.concat([
        df_original_real,      # ~338 Reales originales
        df_new_true,           # 1000 Nuevas Reales
        df_fake_sampled_final  
    ], ignore_index=True)

    # Mapeo y limpieza final del set de entrenamiento
    mapeo_train = {'True': 1, 'Fake': 0} 
    df_train_final['label'] = df_train_final['label_cat'].map(mapeo_train)
    df_train_final = df_train_final.dropna(subset=['label', 'text'])
    df_train_final['label'] = df_train_final['label'].astype(int)


    # 3. --- CONJUNTO DE PRUEBA (TEST: Test.xlsx) ---
    df_test_raw = pd.read_excel(ruta_excel_test, sheet_name=0) 
    df_test_raw.columns = df_test_raw.columns.str.lower() 
    df_test_final = df_test_raw.rename(columns={'text': 'text', 'category': 'label_cat'}) 
    
    # Mapeo corregido para el conjunto de PRUEBA (TRUE/FALSE)
    mapeo_test = {'TRUE': 1, 'FALSE': 0} 
    df_test_final['label_cat'] = df_test_final['label_cat'].astype(str).str.strip().str.upper() 
    df_test_final['label'] = df_test_final['label_cat'].map(mapeo_test)
    
    # Limpieza final del set de prueba
    df_test_final = df_test_final.dropna(subset=['label', 'text'])
    df_test_final['label'] = df_test_final['label'].astype(int)
    
    
    # 4. --- RETORNO DE DATOS ---
    X_train = df_train_final['text']
    y_train = df_train_final['label']
    X_test = df_test_final['text']
    y_test = df_test_final['label']
    
    print(f"Total datos de Entrenamiento (Real: {y_train.sum()}, Falsa: {len(y_train) - y_train.sum()}): {len(X_train)}")
    print(f"Total datos de Prueba (Real: {y_test.sum()}, Falsa: {len(y_test) - y_test.sum()}): {len(X_test)}")
    
    return X_train, X_test, y_train, y_test