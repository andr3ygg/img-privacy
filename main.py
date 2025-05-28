import os
import cv2
import numpy as np
from urllib.request import urlretrieve
from tkinter import filedialog, Tk

def download_model_files(prototxt_path, prototxt_url, model_path, model_url):
    os.makedirs(os.path.dirname(prototxt_path), exist_ok=True)
    if not os.path.isfile(prototxt_path):
        print("Descargando deploy.prototxt...")
        urlretrieve(prototxt_url, prototxt_path)
    if not os.path.isfile(model_path):
        print("Descargando modelo caffemodel...")
        urlretrieve(model_url, model_path)

def select_image_file():
    Tk().withdraw()
    path = filedialog.askopenfilename(
        title='Selecciona una imagen',
        filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
    )
    if not path:
        print("No se seleccionó ninguna imagen.")
        exit()
    return path

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"No se pudo cargar la imagen: {image_path}")
        exit()
    return image

def detect_and_blur_faces(image, net, conf_threshold=0.5, max_img_size=2000):
    (h0, w0) = image.shape[:2]
    image_for_net = image.copy()

    # Redimensionar solo para el modelo si es muy grande
    if max(h0, w0) > max_img_size:
        scale = max_img_size / max(h0, w0)
        image_for_net = cv2.resize(image, (int(w0*scale), int(h0*scale)))
    else:
        scale = 1.0

    (h, w) = image_for_net.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image_for_net, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Escalar las cajas a tamaño original
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w0, x2), min(h0, y2)

            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_width = x2 - x1
            k = max(15, (face_width // 3) | 1)
            blurred_face = cv2.GaussianBlur(face, (k, k), 30)
            image[y1:y2, x1:x2] = blurred_face
    
    # Aplicar blur a todas las caras detectadas
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            (x1, y1, x2, y2) = box.astype("int")
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

    # Mostrar y guardar
    cv2.namedWindow("Caras con blur", cv2.WINDOW_NORMAL)
    cv2.imshow("Caras con blur", image)
    #cv2.resizeWindow("Blurred Image", 800, 600)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return image

def main():
    model_dir = "models"
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"

    prototxt_path = os.path.join(model_dir, "deploy.prototxt")
    model_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

    download_model_files(prototxt_path, prototxt_url, model_path, model_url)
    image_path = select_image_file()
    image = load_image(image_path)

    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    output_image = detect_and_blur_faces(image, net)

    cv2.imshow("Caras con blur", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    name, ext = os.path.splitext(image_path)
    output_path = f"{name}_blurred{ext}"
    cv2.imwrite(output_path, output_image)
    print(f"Imagen guardada en: {output_path}")


if __name__ == "__main__":
    main()
