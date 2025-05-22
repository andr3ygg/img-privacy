import os
import cv2
import numpy as np
from urllib.request import urlretrieve
from tkinter import filedialog, Tk

# Crear carpeta para modelos si no existe
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

# Archivos y URLs
prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

prototxt_path = os.path.join(model_dir, "deploy.prototxt")
model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

# Descargar si no existe
if not os.path.isfile(prototxt_path):
    print("Descargando deploy.prototxt...")
    urlretrieve(prototxt_url, prototxt_path)

if not os.path.isfile(model_path):
    print("Descargando modelo caffemodel...")
    urlretrieve(model_url, model_path)

# Selección de imagen
Tk().withdraw()
image_path = filedialog.askopenfilename(title='Selecciona una imagen', filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
if not image_path:
    print("No se seleccionó ninguna imagen.")
    exit()

# Cargar imagen
image = cv2.imread(image_path)
(h, w) = image.shape[:2]

# Cargar modelo DNN
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Preparar la imagen para el modelo
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()

# Aplicar blur a todas las caras detectadas
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = image[y1:y2, x1:x2]
        blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
        image[y1:y2, x1:x2] = blurred_face

# Mostrar y guardar
cv2.imshow("Caras con blur", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_path = image_path.replace(".", "_blurred.")
cv2.imwrite(output_path, image)
print(f"Imagen guardada en: {output_path}")
