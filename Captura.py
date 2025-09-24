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

