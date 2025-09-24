<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>Captura de Fotos para {{ usuario.nombre_completo }}</title>
  <style>
    body { font-family: Arial, sans-serif; background: #f2f2f2; text-align: center; }
    header { background: #00008B; color: white; padding: 20px; }
    #camera-container { width: 640px; margin: auto; background: #333; border-radius: 10px; overflow: hidden; }
    video { width: 100%; }
    .btn { padding: 10px 20px; margin: 10px; border: none; border-radius: 5px; cursor: pointer; }
    .btn-capturar { background: #28a745; color: white; }
    .btn-volver { background: #00008B; color: white; }
    #canvas { display: none; }
  </style>
</head>
<body>
  <header>
    <h1>Captura de Fotos para {{ usuario.nombre_completo }}</h1>
  </header>
  <div id="camera-container">
    <video id="video" autoplay></video>
  </div>
  <div id="controls">
    <button class="btn btn-capturar" id="captureBtn">Capturar Foto</button>
    <button class="btn btn-volver" onclick="location.href='{{ url_for('index') }}'">Volver</button>
  </div>
  <!-- Canvas oculto para capturar la imagen -->
  <canvas id="canvas"></canvas>
  <!-- Formulario para enviar la imagen capturada -->
  <form id="photoForm" method="POST">
    <input type="hidden" name="image_data" id="image_data">
  </form>
  <script>
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('captureBtn');
    const image_data_input = document.getElementById('image_data');
    const photoForm = document.getElementById('photoForm');

    // Solicitar acceso a la cámara
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => { video.srcObject = stream; })
      .catch(err => { console.error("Error accediendo a la cámara:", err); });
    // Capturar la foto y enviarla mediante el formulario
    captureBtn.addEventListener('click', () => {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const context = canvas.getContext('2d');
      context.drawImage(video, 0, 0);
      const imageDataURL = canvas.toDataURL('image/png');
      image_data_input.value = imageDataURL;
      photoForm.submit();
    });
  </script>
</body>
</html>

Anexo D
from flask import Flask, render_template, request, redirect, url_for, jsonify, Response, make_response
from flask_sqlalchemy import SQLAlchemy
import os
import numpy as np
import joblib
import base64
from datetime import datetime
from picamera2 import Picamera2
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from skimage.feature import hog
import cv2
import shutil
import subprocess
import csv
from io import StringIO
import logging
import sqlite3
import threading
import gpiod
from collections import defaultdict
import time

# Carpeta base para guardar imagenes (entrenamiento)
path_default = "/home/teleco/Documentos/entrenamiento/"

app = Flask(__name__)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'tu_base_de_datos.db')}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db = SQLAlchemy(app)

class camera:
    lock_camera = threading.RLock()
    def __init__(self):
        self.pause_event = threading.Event()
        self.pause_event.clear()  # por defecto no estÃ¡ en pausa
        # ConfiguraciÃ³n de logging
        logging.basicConfig(
            filename='/home/teleco/Documentos/logs/predictor.log',
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=logging.INFO
        )

        # Ruta al archivo Haar Cascade para detecciÃ³n de ojos
        self.eye_cascade = cv2.CascadeClassifier('/home/teleco/Documentos/code/eyecascade.xml')

        # Ruta del modelo previamente entrenado
        self.modelo_path = "/home/teleco/Documentos/code/modelo_iris_ojos.pkl"

        # Verificar si el modelo existe
        if not joblib.os.path.exists(self.modelo_path):
            raise FileNotFoundError(f"- No se encontrÃ³ el modelo en '{self.modelo_path}'.")

        # Cargar el modelo entrenado
        self.svm = joblib.load(self.modelo_path)

        # Listado de personas segÃºn las etiquetas del modelo
        conn = sqlite3.connect("/home/teleco/Documentos/web_iris/instance/tu_base_de_datos.db")
        cursor = conn.cursor()
        cursor.execute("SELECT nombre_completo FROM Usuario;")
        registros = cursor.fetchall()
        self.carpetas = [fila[0] for fila in registros]
        self.carpetas.sort()

        # TamaÃ±o de la imagen usado para entrenar el modelo (64x64)
        self.TAMANO_IMAGEN = (64, 64)

        # ConfiguraciÃ³n de GPIO con gpiod
        self.chip = gpiod.Chip("gpiochip0")  # Ajustar si es necesario
        self.line = self.chip.get_line(17)
        self.line.request(consumer="eye_detector", type=gpiod.LINE_REQ_DIR_OUT)

        self.deteccion_tiempos = defaultdict(lambda: 0)
        self.umbral_tiempo = 8  # Segundos necesarios de detecciÃ³n estable
        self.ultima_deteccion = None

        self.cam  = None
        self.get_picamera()
        self.prediction_obj = threading.Thread(name='prediction_thread', target=self.__read_thread, daemon=True)
        self.prediction_obj.start()

    def get_picamera(self):
        if self.cam is None:
            try:
                # Configuracion de la camara con Picamera2
                self.cam = Picamera2()
                self.cam.configure(self.cam.create_video_configuration(raw={"size":(1640,1232)}, main={"format":'RGB888',"size": (640,480)}))
                self.cam.start()
                return True

            except Exception as e:
                print(f"No se pudo iniciar la cÃ¡mara: {e}")
                return False
        else:
            return True

    def release_picamera(self):
        if self.cam != None:
            self.cam.stop()
        self.cam  = None

    def registrar_acceso(self, nombre):
        with sqlite3.connect("/home/teleco/Documentos/web_iris/instance/tu_base_de_datos.db") as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO accesos_aula (nombre, fecha) VALUES (?, datetime('now'))", (nombre,))
            conn.commit()

    def predecir_ojos(self, frame, x, y, w, h):
        ojo = frame[y:y+h, x:x+w]
        ojo = cv2.GaussianBlur(ojo, (5, 5), 0)
        ojo_resized = cv2.resize(ojo, self.TAMANO_IMAGEN)
        ojo_resized = cv2.equalizeHist(ojo_resized)
        hog_features = hog(ojo_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
        
        if len(hog_features) != self.svm.n_features_in_:
            raise ValueError(f"- El nÃºmero de caracterÃ­sticas es incorrecto. Se esperaban {self.svm.n_features_in_}, pero se obtuvieron {len(hog_features)}.")
        
        probabilidades = self.svm.predict_proba([hog_features])[0]
        prediccion = np.argmax(probabilidades)
        confianza = probabilidades[prediccion]
        if confianza > 0.8:
            return self.carpetas[prediccion], confianza
        return None, None

    def acquire_camera(self):
        self.lock_camera.acquire()

    def release_camera(self):
        try:
            self.lock_camera.release()
        except:
            pass

    # Funcion para entrenar el modelo de iris (deteccion de ojos)
    def entrenar_modelo_iris(self):
        salida = ""
        
        # Lista de carpetas; ajusta segun tus datos
        self.carpetas = [usuario.nombre_completo for usuario in db.session.query(Usuario).all()]
        self.carpetas.sort()

        data = []
        labels = []

        # Recorrer cada carpeta y extraer caracteristicas HOG de los ojos
        for label, carpeta in enumerate(self.carpetas):
            carpeta_path = os.path.join(path_default, carpeta)
            if not os.path.exists(carpeta_path):
                salida += f"La carpeta {carpeta_path} no existe.\n"
                logging.info('- ERROR: La carpeta %s no existe.\r', carpeta_path)
                continue
            for imagen_nombre in os.listdir(carpeta_path):
                imagen_path = os.path.join(carpeta_path, imagen_nombre)
                img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                ojos = self.eye_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                for (x, y, w, h) in ojos:
                    ojo = img[y:y+h, x:x+w]
                    try:
                        ojo_resized = cv2.resize(ojo, (64, 64))
                    except Exception as e:
                        salida += f"Error al redimensionar {imagen_path}: {e}\n"
                        logging.info('- ERROR: No su pudo redimensionar %s (error = %s).\r', imagen_path, e)
                        continue
                    hog_features = hog(ojo_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
                    data.append(hog_features)
                    labels.append(label)
                    break  # Solo se toma el primer ojo detectado
        
        if len(data) == 0:
            return "No se encontraron datos para entrenar."

        data = np.array(data)
        labels = np.array(labels)

        # Si solo hay una clase, utilizamos One-Class SVM
        if len(set(labels)) < 2:
            salida += "Se detecto solo una clase. Usando One-Class SVM para entrenamiento.\n"
            from sklearn.svm import OneClassSVM
            svm = OneClassSVM(kernel='linear', nu=0.1)
            try:
                svm.fit(data)
            except Exception as e:
                return f"Error en el entrenamiento con One-Class SVM: {e}"
            # En One-Class SVM la evaluacion se hace de otra forma (por ejemplo, comprobando la capacidad de detectar outliers)
            try:
                joblib.dump(svm, self.modelo_path)
                logging.info('- ENTRENAMIENTO: Modelo guardado satisfactoriamente')
                salida += "Modelo One-Class SVM entrenado exitosamente.\n"
            except Exception as e:
                salida += f"Error al guardar el modelo: {e}\n"
        else:
            from sklearn.svm import SVC
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
            svm = SVC(kernel='linear', probability=True)
            try:
                svm.fit(X_train, y_train)
            except Exception as e:
                return f"Error en el entrenamiento con SVC: {e}"
            y_pred = svm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            salida += f"Precision del modelo: {accuracy:.2f}\n"
            # Guardar el modelo entrenado
            try:
                joblib.dump(svm, self.modelo_path)
                logging.info('- ENTRENAMIENTO: Modelo guardado satisfactoriamente')
                #salida += f"Modelo entrenado y guardado en '{self.modelo_path}'.\n"
            except Exception as e:
                salida += f"Error al guardar el modelo: {e}\n"

        # Funcion de prueba para predecir en una imagen
        def predecir_ojos(imagen_path):
            img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return "No se pudo leer la imagen de prueba."
            ojos = self.eye_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x, y, w, h) in ojos:
                ojo = img[y:y+h, x:x+w]
                try:
                    ojo_resized = cv2.resize(ojo, (64, 64))
                except Exception as e:
                    return f"Error al redimensionar la imagen de prueba: {e}"
                hog_features = hog(ojo_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
                # En One-Class SVM se obtiene un score (la prediccion sera 1 si es "normal")
                prediccion = svm.predict([hog_features])[0]
                try:
                    return f"Ojo detectado en la prueba, pertenece a {self.carpetas[prediccion]}"
                except:
                    return "Ojo detectado no coincide con la clase esperada."
            return "No se detecto ningun ojo en la imagen de prueba."

        def obtener_imagen_aleatoria(ruta_base):
            import random
            extensiones_validas = ('.jpg', '.jpeg', '.png', '.bmp')
            imagenes = []

            for root, dirs, files in os.walk(ruta_base):
                for file in files:
                    if file.lower().endswith(extensiones_validas):
                        imagenes.append(os.path.join(root, file))
            
            if not imagenes:
                return None  # No se encontraron imÃ¡genes
            
            return random.choice(imagenes)

        # Ruta de la imagen de prueba (modificala segun corresponda)
        prueba_path = obtener_imagen_aleatoria("/home/teleco/Documentos/prueba_fotos/")
        resultado_prediccion = predecir_ojos(prueba_path)
        salida += resultado_prediccion

        logging.info('- ENTRENAMIENTO: %s', salida)
        return salida

    def pause_prediction(self):
        self.pause_event.set()
        logging.info(">> Pausando predicciÃ³n.")

    def resume_prediction(self):
        self.pause_event.clear()
        logging.info(">> Reanudando predicciÃ³n.")

    #=======================================#
    #       Hilo de lectura         		#
    #=======================================#
    def __read_thread(self):
        logging.info("- Modelo cargado exitosamente.")
        start_time = time.time()
        stop_time = start_time
        while self.cam is None and (stop_time-start_time) < 30:
            self.get_picamera()
            time.sleep(1)
        stop_time = time.time()

        if self.cam is not None:
            tiempo_actual = time.time()
            self.deteccion_tiempos = {}

            while True:
                if self.pause_event.is_set():
                    time.sleep(0.1)
                    continue
                try:
                    self.acquire_camera()
                    frame = self.cam.capture_array()
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    ojos = self.eye_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=10, minSize=(30, 30))

                    detectado = None
                    for (x, y, w, h) in ojos:
                        try:
                            persona, confianza = self.predecir_ojos(gray, x, y, w, h)
                            #print(f"persona: {persona}, confianza: {confianza}")
                            if persona and confianza > 0.9:
                                detectado = persona
                                break
                            #else:
                            #    logging.warning('- WARNING: Persona desconocida')
                        except Exception as e:
                            logging.warning(f"- Error al predecir: {e}")
                    
                    if detectado:
                        if detectado == self.ultima_deteccion:
                            self.deteccion_tiempos[detectado] += time.time() - tiempo_actual
                        else:
                            self.deteccion_tiempos[detectado] = 0
                            self.ultima_deteccion = detectado

                        if self.deteccion_tiempos[detectado] >= self.umbral_tiempo:
                            logging.info(f"- {detectado} detectado por {self.umbral_tiempo} segundos. Activando GPIO 17.")
                            self.registrar_acceso(detectado)
                            self.line.set_value(1)
                            time.sleep(5)  # GPIO activo 5 segundos
                            self.line.set_value(0)
                            self.deteccion_tiempos[detectado] = 0  # Reinicia tiempo
                    else:
                        tiempo_actual = time.time()

                    time.sleep(0.1)  # Descanso para no saturar CPU
                    self.release_camera()
                except Exception as e:
                    logging.exception("- Error dentro del bucle principal (error = %s)", e)
                    self.release_camera()
        else:
            logging.exception("- ERROR: No se pudo iniciar la camara")
        if self.line:
            self.line.release()
        logging.info('- CAMERA: read process terminated.\r')
            

# Modelo de Usuario
class Usuario(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre_completo = db.Column(db.String(100), nullable=False)
    cedula = db.Column(db.String(20), nullable=False)
    asignatura = db.Column(db.String(50), nullable=False)
    

    def __init__(self, nombre_completo, cedula, asignatura):
        self.nombre_completo = nombre_completo
        self.cedula = cedula
        self.asignatura = asignatura

class UsuarioEliminado(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nombre_completo = db.Column(db.String(100), nullable=False)
    cedula = db.Column(db.String(20), nullable=False)
    asignatura = db.Column(db.String(50), nullable=False)
    fecha_eliminacion = db.Column(db.DateTime, default=datetime.utcnow)

    def __init__(self, nombre_completo, cedula, asignatura):
        self.nombre_completo = nombre_completo
        self.cedula = cedula
        self.asignatura = asignatura

class accesos_aula(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fecha = db.Column(db.DateTime, default=datetime.utcnow)
    nombre = db.Column(db.String(100), nullable=False)

    def __init__(self, nombre):
        self.nombre = nombre

# Ruta principal: Registro, eliminacion y listado de usuarios
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'agregar' in request.form:
            nombre_completo = request.form.get('nombre_completo')
            cedula = request.form.get('cedula')
            asignatura = request.form.get('asignatura')
            nuevo_usuario = Usuario(nombre_completo=nombre_completo, cedula=cedula, asignatura=asignatura)
            db.session.add(nuevo_usuario)
            db.session.commit()
            # Redirige a la pagina para capturar la foto, pasando el id del usuario recien creado
            return redirect(url_for('tomarfotos_usuario', usuario_id=nuevo_usuario.id))
    usuarios = Usuario.query.all()

    picam.release_camera()
    picam.resume_prediction()
    # Iniciando proceso predictor
    #try:
    #    subprocess.call(["sudo", "supervisorctl", "start", "predictor"])
    #except Exception as e:
    #    print(f"Error al iniciar predictor: {e}")

    return render_template('index.html', usuarios=usuarios)
    
# Ruta para capturar fotos para un usuario especifico
@app.route('/tomarfotos_usuario/<int:usuario_id>', methods=['GET', 'POST'])
def tomarfotos_usuario(usuario_id):
    usuario = Usuario.query.get(usuario_id)
    if not usuario:
        return redirect(url_for('index'))

    # Detener proceso principal
    picam.pause_prediction()
    #try:
    #    subprocess.call(["sudo", "supervisorctl", "stop", "predictor"])
    #except Exception as e:
    #    print(f"Error al detener el proceso de predicciÃ³n: {e}")
    # Utiliza path_default y el nombre del usuario para crear la carpeta de entrenamiento
    path_user = os.path.join(path_default, usuario.nombre_completo)
    os.makedirs(path_user, exist_ok=True)
    return render_template('tomarfotos_usuario.html', usuario=usuario)

# Funcion generadora para el streaming de video
def generate_frames():
    while True:
        if picam.cam is not None:
            frame = picam.cam.capture_array()
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# Ruta para el video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Ruta para Capturar y Guardar una Foto
@app.route('/capturar_foto', methods=['POST'])
def capturar_foto():
    usuario_id = request.json.get("usuario_id")
    if not usuario_id:
        return jsonify({"error": "Falta usuario_id"}), 400
    
    usuario = Usuario.query.get(usuario_id)
    if not usuario:
        return jsonify({"error": "Usuario no encontrado"}), 404

    # Capturar imagen
    if picam.cam is not None:
        frame = picam.cam.capture_array()
    else:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', frame)
    
    # Guardar imagen en la carpeta de entrenamiento (usando el nombre del usuario)
    ruta_usuario = os.path.join(path_default, usuario.nombre_completo)
    os.makedirs(ruta_usuario, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_path = os.path.join(ruta_usuario, f"{timestamp}.jpg")
    with open(image_path, "wb") as f:
        f.write(buffer)
    
    return jsonify({"mensaje": "Foto capturada exitosamente", "ruta": image_path})

# Ruta para Eliminar un Usuario
@app.route('/eliminar_usuario', methods=['POST'])
def eliminar_usuario():
    import shutil
    usuario_id = request.json.get("id")
    
    if not usuario_id:
        return jsonify({"error": "ID de usuario no proporcionado"}), 400
    
    usuario = db.session.get(Usuario, usuario_id)
    if usuario:
        try:
            # Guardar datos en la nueva tabla antes de eliminar
            usuario_eliminado = UsuarioEliminado(
                nombre_completo=usuario.nombre_completo,
                cedula=usuario.cedula,
                asignatura=usuario.asignatura
            )
            db.session.add(usuario_eliminado)

            # Eliminar carpeta de imÃ¡genes
            ruta_usuario = os.path.join(path_default, usuario.nombre_completo)
            shutil.rmtree(ruta_usuario)

            # Eliminar de la tabla principal
            db.session.delete(usuario)
            db.session.commit()
            return jsonify({"mensaje": "Usuario eliminado correctamente"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500 
    else:
        return jsonify({"error": "Usuario no encontrado"}), 404

# Ruta para entrenar el modelo (se activa al pulsar el boton en la web)
@app.route('/entrenar_modelo', methods=['POST'])
def ruta_entrenar_modelo():
    picam.pause_prediction()
    try:
        resultado = picam.entrenar_modelo_iris()
        picam.resume_prediction()
        try:
            resultado = resultado.split(':')[1].strip()
            return jsonify({"mensaje": "Entrenamiento completado", "precision": resultado})
        except:
            return jsonify({"mensaje": "Entrenamiento completado", "precision": resultado})
    except Exception as e:
        return jsonify({"mensaje": "Error durante el entrenamiento", "error": str(e)}), 500

@app.route('/descargar_usuarios_eliminados')
def descargar_usuarios_eliminados():
    usuarios = UsuarioEliminado.query.all()
    
    # Crear CSV en memoria
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Nombre Completo', 'CC', 'Asignatura', 'Fecha Eliminado'])

    for u in usuarios:
        writer.writerow([u.nombre_completo, u.cedula, u.asignatura, u.fecha_eliminacion])

    # Preparar respuesta como archivo descargable
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=usuarios_eliminados.csv"
    response.headers["Content-type"] = "text/csv"
    return response

@app.route('/descargar_AccesoAula')
def descargar_AccesoAula():
    usuarios = accesos_aula.query.all()
    
    # Crear CSV en memoria
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Nombre', 'Fecha Acceso'])

    for u in usuarios:
        writer.writerow([u.nombre, u.fecha])

    # Preparar respuesta como archivo descargable
    response = make_response(output.getvalue())
    response.headers["Content-Disposition"] = "attachment; filename=Reporte_acceso.csv"
    response.headers["Content-type"] = "text/csv"
    return response

picam = camera()
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=80
