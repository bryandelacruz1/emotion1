import os
import base64
import cv2
import numpy as np
import mediapipe as mp
import matplotlib
 # Desactivar el backend interactivo de Tkinter
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify, send_from_directory
from io import BytesIO
from werkzeug.utils import secure_filename

# Configuración del servidor Flask
app = Flask(__name__)
UPLOAD_FOLDER = '/static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Emociones en el modelo FER+
EMOTION_ORDER = ["Neutral", "Feliz", "Sorpresa", "Triste", "Enojado", "Disgustado", "Miedo", "Desprecio"]

# Validación de archivos permitidos
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Función de análisis facial con detección de emociones
def analyze_face(image_path):
    try:
        # Cargar MediaPipe FaceMesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

        # Cargar modelo ONNX de emociones
        emotion_model_path = 'emotion-ferplus-8.onnx'
        if not os.path.exists(emotion_model_path):
            raise FileNotFoundError(f"Modelo no encontrado: {emotion_model_path}")

        emotion_model = cv2.dnn.readNetFromONNX(emotion_model_path)

        # Cargar imagen
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("No se pudo cargar la imagen.")

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = gray_image.shape

        # Detectar puntos faciales
        results = face_mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            raise ValueError("No se detectó ningún rostro.")

        # Obtener coordenadas de la cara
        face_landmarks = results.multi_face_landmarks[0]
        x_min = int(min([lm.x for lm in face_landmarks.landmark]) * width)
        y_min = int(min([lm.y for lm in face_landmarks.landmark]) * height)
        x_max = int(max([lm.x for lm in face_landmarks.landmark]) * width)
        y_max = int(max([lm.y for lm in face_landmarks.landmark]) * height)

        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(width, x_max), min(height, y_max)

        face_region = gray_image[y_min:y_max, x_min:x_max]
        if face_region.size == 0:
            raise ValueError("No se pudo extraer la región del rostro.")

        # Preprocesamiento de la imagen para el modelo
        face_resized = cv2.resize(face_region, (64, 64))  # Asegurar tamaño 64x64
        face_resized = face_resized.astype("float32") / 255.0  # Normalización
        face_resized = np.expand_dims(face_resized, axis=0)  # Agregar dimensión batch
        face_resized = np.expand_dims(face_resized, axis=0)  # Agregar canal (1, 64, 64)

        # Enviar imagen al modelo
        blob = cv2.dnn.blobFromImage(face_resized, scalefactor=1.0, size=(64, 64), mean=(0, 0, 0), swapRB=False, crop=False)
        emotion_model.setInput(blob)
        emotion_predictions = emotion_model.forward()

        # Obtener la emoción con mayor probabilidad
        detected_emotion = EMOTION_ORDER[np.argmax(emotion_predictions)]

        # Dibujar puntos faciales en la imagen
        keypoint_image = image.copy()
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * width), int(lm.y * height)
            cv2.circle(keypoint_image, (x, y), 2, (0, 255, 0), -1)

        _, buffer = cv2.imencode('.png', keypoint_image)
        keypoint_image_base64 = base64.b64encode(buffer).decode('utf-8')

        # Transformaciones visuales
        transformations = [
            ("Original", gray_image),
            ("Volteado horizontal", cv2.flip(gray_image, 1)),
            ("Brillo aumentado", cv2.convertScaleAbs(gray_image, alpha=1.2, beta=50)),
            ("Invertido vertical", cv2.flip(gray_image, 0))
        ]

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        for ax, (title, img) in zip(axes.flatten(), transformations):
            ax.imshow(img, cmap='gray')
            ax.set_title(title)
            ax.axis('off')

        buf = BytesIO()
        plt.tight_layout()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        transformations_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return keypoint_image_base64, transformations_base64, detected_emotion

    except Exception as e:
        raise RuntimeError(f"Error en analyze_face: {e}")

# Ruta de análisis
@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'existing_file' in request.form:
            filename = request.form['existing_file']
        elif 'file' in request.files:
            file = request.files['file']
            if file.filename == '' or not allowed_file(file.filename):
                return jsonify({'error': 'Archivo inválido'}), 400

            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            return jsonify({'error': 'No se proporcionó archivo'}), 400

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        keypoint_image, transformations_image, detected_emotion = analyze_face(filepath)

        return jsonify({
            'success': True,
            'keypoint_image': keypoint_image,
            'transformations_image': transformations_image,
            'emotion': detected_emotion
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
