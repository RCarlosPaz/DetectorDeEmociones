%%writefile app.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import os
import pandas as pd  # Importar pandas
import seaborn as sns # Importar seaborn

# --- Configuración de la Página de Streamlit ---
st.set_page_config(
    page_title="Detector de Emociones Faciales",
    page_icon="😊",
    layout="centered",
    initial_sidebar_state="auto",
)

# --- Título y Descripción ---
st.title("😊 Detector de Emociones Faciales")
st.markdown(
    "Bienvenido al detector de emociones faciales. "
    "Sube una imagen y te diré la emoción predominante en el rostro."
)

# --- Definir etiquetas de emociones (como en tu notebook original) ---
emotion_labels = ['Ira', 'Disgusto', 'Miedo', 'Felicidad', 'Tristeza', 'Sorpresa', 'Neutral']

# --- Función para crear el modelo (como en tu notebook original) ---
@st.cache_resource
def create_emotion_model(input_shape=(48, 48, 1), num_classes=len(emotion_labels)):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Compile el modelo (necesario incluso si carga pesos)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# --- Cargar el modelo y los pesos ---
# NOTA: Asegúrate de que 'emotion_model_weights.h5' esté disponible en tu entorno de Streamlit Cloud.
# Puedes subirlo a tu repositorio de GitHub junto con este archivo.
# Alternativamente, puedes descargarlo de una URL pública si es muy grande.

model = create_emotion_model()

try:
    # Intenta cargar los pesos del modelo.
    # Si no tienes un archivo .h5, puedes omitir esta línea para usar un modelo sin entrenar,
    # pero las predicciones no serán precisas.
    model_weights_path = 'emotion_model_weights.h5'
    if os.path.exists(model_weights_path):
        model.load_weights(model_weights_path)
        st.success("Pesos del modelo cargados exitosamente.")
    else:
        st.warning("Archivo de pesos del modelo 'emotion_model_weights.h5' no encontrado. "
                   "Las predicciones no serán precisas. "
                   "Por favor, asegúrate de subir los pesos o entrenar el modelo.")
except Exception as e:
    st.error(f"Error al cargar los pesos del modelo: {e}")
    st.warning("Las predicciones se realizarán con un modelo sin entrenar.")

# --- Cargar clasificador Haar Cascade para detección de rostros ---
@st.cache_resource
def load_face_cascade():
    # Descargar el clasificador Haar Cascade si no existe
    cascade_path = 'haarcascade_frontalface_default.xml'
    if not os.path.exists(cascade_path):
        import requests
        url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        r = requests.get(url, allow_redirects=True)
        with open(cascade_path, 'wb') as f:
            f.write(r.content)
    return cv2.CascadeClassifier(cascade_path)

face_cascade = load_face_cascade()

# --- Función de detección y preprocesamiento de rostros ---
def detect_and_preprocess_face(image_bytes, target_size=(48, 48)):
    # Convertir bytes a array de numpy
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Error: No se pudo decodificar la imagen.")
        return None, None, None

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None, None, img # Devuelve la imagen original para mostrar sin rostros detectados

    # Tomar el primer rostro detectado
    (x, y, w, h) = faces[0]
    face_roi = gray_img[y:y+h, x:x+w]

    # Redimensionar y normalizar
    processed_face = cv2.resize(face_roi, target_size)
    processed_face = processed_face.astype('float32') / 255.0
    processed_face = np.expand_dims(processed_face, axis=-1)  # Añadir dimensión de canal
    processed_face = np.expand_dims(processed_face, axis=0)   # Añadir dimensión de lote

    return processed_face, faces[0], img # También devolvemos la imagen original y la caja del rostro

# --- Función para mostrar la predicción en Streamlit ---
def display_prediction_streamlit(original_img, face_coords, emotion_probabilities, emotion_labels):
    if face_coords is not None:
        (x, y, w, h) = face_coords

        # Dibujar un rectángulo alrededor del rostro
        cv2.rectangle(original_img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Obtener la emoción predicha con mayor probabilidad
        predicted_emotion_idx = np.argmax(emotion_probabilities)
        predicted_emotion = emotion_labels[predicted_emotion_idx]
        confidence = emotion_probabilities[0][predicted_emotion_idx] * 100

        # Mostrar la emoción y la confianza en la imagen
        text = f"{predicted_emotion}: {confidence:.2f}%"
        cv2.putText(original_img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        st.subheader("Resultado de la Detección")
        # Convertir a formato RGB para mostrar en Streamlit
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="Rostro Detectado y Emoción Predicha", use_column_width=True)

        st.markdown(f"### Emoción Predominante: **{predicted_emotion}** con {confidence:.2f}% de confianza.")

        st.subheader("Probabilidades de Emoción:")
        # Crear un DataFrame para visualizar las probabilidades
        df_probabilities = pd.DataFrame({"Emoción": emotion_labels, "Probabilidad": emotion_probabilities[0]})
        df_probabilities = df_probabilities.sort_values(by="Probabilidad", ascending=False)
        
        # Mostrar las probabilidades en un gráfico de barras
        fig_probs = plt.figure(figsize=(8, 4))
        sns.barplot(x="Probabilidad", y="Emoción", data=df_probabilities, palette="viridis")
        plt.title("Distribución de Probabilidades de Emoción")
        plt.xlabel("Probabilidad")
        plt.ylabel("Emoción")
        plt.xlim(0, 1) # Asegurar que el eje x va de 0 a 1
        st.pyplot(fig_probs)

    else:
        st.warning("No se detectaron rostros en la imagen. Por favor, sube una imagen clara con un rostro.")
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="Imagen Original (sin rostros detectados)", use_column_width=True)

# --- Interfaz de Usuario de Streamlit ---
uploaded_file = st.file_uploader("Sube una imagen de un rostro aquí:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer la imagen como bytes
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption="Imagen Subida", use_column_width=True)

    # Botón para iniciar la predicción
    if st.button("Detectar Emoción"):
        with st.spinner("Detectando rostro y prediciendo emoción..."):
            processed_face, face_coords, original_img = detect_and_preprocess_face(image_bytes)

            if processed_face is not None:
                # Realizar la predicción
                emotion_probabilities = model.predict(processed_face)
                display_prediction_streamlit(original_img, face_coords, emotion_probabilities, emotion_labels)
            else:
                if original_img is not None:
                    st.warning("No se detectaron rostros en la imagen. Por favor, sube una imagen clara con un rostro.")
                    st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), caption="Imagen Original (sin rostros detectados)", use_column_width=True)
                else:
                     st.error("Hubo un problema al procesar la imagen.")


# --- Diseño / Estilo (Opcional: puedes personalizarlo con CSS) ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f0f2f6; /* Un fondo gris claro */
    }
    .stButton>button {
        background-color: #4CAF50; /* Verde */
        color: white;
        font-size: 1.2em;
        padding: 0.5em 1em;
        border-radius: 10px;
        border: 2px solid #388E3C; /* Verde oscuro */
    }
    .stButton>button:hover {
        background-color: #45a049;
        border-color: #2E7D32;
    }
    .stFileUploader {
        background-color: #e0e0e0;
        border-radius: 15px;
        padding: 1em;
        border: 1px dashed #9e9e9e;
    }
    .stFileUploader > div > div > button {
        background-color: #2196F3; /* Azul */
        color: white;
        border-radius: 8px;
        border: none;
    }
    .stFileUploader > div > div > button:hover {
        background-color: #1976D2;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #3f51b5; /* Un azul más oscuro */
    }
    .css-1d3w5sm, .css-1dp1f07, .css-1y48gtm {
        border-radius: 15px; /* Bordes redondeados para contenedores */
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1); /* Sutil sombra */
        padding: 15px;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True
)
