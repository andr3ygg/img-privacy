import os
import cv2
import numpy as np
from urllib.request import urlretrieve
from tkinter import filedialog, Tk, Button, Label, messagebox, Canvas
from PIL import Image, ImageTk


def download_model_files(prototxt_path, prototxt_url, model_path, model_url):
    os.makedirs(os.path.dirname(prototxt_path), exist_ok=True)
    if not os.path.isfile(prototxt_path):
        print("Descargando deploy.prototxt...")
        urlretrieve(prototxt_url, prototxt_path)
    if not os.path.isfile(model_path):
        print("Descargando modelo caffemodel...")
        urlretrieve(model_url, model_path)


def detect_and_blur_faces(image, net, conf_threshold=0.5, max_img_size=2000):
    (h0, w0) = image.shape[:2]
    image_for_net = image.copy()

    if max(h0, w0) > max_img_size:
        scale = max_img_size / max(h0, w0)
        image_for_net = cv2.resize(image, (int(w0 * scale), int(h0 * scale)))
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
            x1, y1 = int(x1 / scale), int(y1 / scale)
            x2, y2 = int(x2 / scale), int(y2 / scale)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w0, x2), min(h0, y2)

            face = image[y1:y2, x1:x2]
            if face.size == 0:
                continue
            face_area = (x2 - x1) * (y2 - y1)
            k = max(15, int(np.sqrt(face_area) // 2) | 1)
            blurred_face = cv2.GaussianBlur(face, (k, k), 30)
            image[y1:y2, x1:x2] = blurred_face

    return image


class FaceBlurringApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector y Desenfoque de Caras")

        self.model_dir = "models"
        self.prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        self.model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        self.prototxt_path = os.path.join(self.model_dir, "deploy.prototxt")
        self.model_path = os.path.join(self.model_dir, "res10_300x300_ssd_iter_140000.caffemodel")

        download_model_files(self.prototxt_path, self.prototxt_url, self.model_path, self.model_url)
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_path)

        self.image_path = None
        self.original_image = None
        self.processed_image = None
        self.showing_original = True

        self.manual_blur_regions = []

        self.label = Label(root, text="Selecciona una imagen")
        self.label.pack()

        self.canvas = Canvas(root, width=500, height=400, cursor="cross")
        self.canvas.pack()

        Button(root, text="Cargar Imagen", command=self.load_image).pack(pady=2)
        Button(root, text="Aplicar Blur a Caras", command=self.apply_blur).pack(pady=2)
        Button(root, text="Comparar Antes/Después", command=self.toggle_view).pack(pady=2)
        Button(root, text="Deshacer Blur", command=self.undo_blur).pack(pady=2)
        Button(root, text="Guardar Imagen", command=self.save_image).pack(pady=2)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.rect = None
        self.start_x = self.start_y = 0

    def load_image(self):
        path = filedialog.askopenfilename(
            title='Selecciona una imagen',
            filetypes=[("Image files", "*.jpg;*.jpeg;*.png")]
        )
        if path:
            self.image_path = path
            self.original_image = cv2.imread(path)
            self.processed_image = None
            self.manual_blur_regions.clear()
            self.showing_original = True
            self.display_image(self.original_image)
        else:
            messagebox.showinfo("Info", "No se seleccionó ninguna imagen.")

    def display_image(self, cv2_image):
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        pil_image.thumbnail((500, 400))
        self.tk_image = ImageTk.PhotoImage(pil_image)
        self.canvas.delete("all")
        self.canvas.create_image(250, 200, image=self.tk_image)

    def apply_blur(self):
        if self.original_image is not None:
            self.processed_image = self.original_image.copy()
            self.processed_image = detect_and_blur_faces(self.processed_image, self.net)
            self.manual_blur_regions.clear()
            self.showing_original = False
            self.display_image(self.processed_image)
        else:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")

    def toggle_view(self):
        if self.original_image is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return

        if self.processed_image is None:
            messagebox.showinfo("Info", "Primero aplica el blur.")
            return

        self.showing_original = not self.showing_original
        img = self.original_image if self.showing_original else self.processed_image
        self.display_image(img)

    def undo_blur(self):
        if self.original_image is not None:
            self.processed_image = None
            self.manual_blur_regions.clear()
            self.showing_original = True
            self.display_image(self.original_image)
            messagebox.showinfo("Info", "Se ha restaurado la imagen original.")
        else:
            messagebox.showwarning("Advertencia", "No hay imagen cargada.")

    def save_image(self):
        if self.processed_image is not None:
            name, ext = os.path.splitext(self.image_path)
            output_path = f"{name}_blurred{ext}"
            cv2.imwrite(output_path, self.processed_image)
            messagebox.showinfo("Guardado", f"Imagen guardada en: {output_path}")
        else:
            messagebox.showwarning("Advertencia", "No hay imagen procesada para guardar.")

    # ----- Funciones de selección manual -----
    def on_mouse_down(self, event):
        if self.original_image is None:
            return
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_mouse_drag(self, event):
        if self.rect:
            self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if self.rect:
            x1, y1, x2, y2 = self.canvas.coords(self.rect)
            self.manual_blur_regions.append((x1, y1, x2, y2))
            self.apply_manual_blur_to_image()

    def apply_manual_blur_to_image(self):
        if self.original_image is None:
            return

        img = self.original_image.copy() if self.processed_image is None else self.processed_image.copy()
        h_img, w_img = img.shape[:2]
        scale_x = w_img / 500
        scale_y = h_img / 400

        for x1, y1, x2, y2 in self.manual_blur_regions:
            ix1, iy1 = int(min(x1, x2) * scale_x), int(min(y1, y2) * scale_y)
            ix2, iy2 = int(max(x1, x2) * scale_x), int(max(y1, y2) * scale_y)
            roi = img[iy1:iy2, ix1:ix2]
            if roi.size > 0:
                k = max(15, (min(ix2 - ix1, iy2 - iy1) // 2) | 1)
                img[iy1:iy2, ix1:ix2] = cv2.GaussianBlur(roi, (k, k), 30)

        self.processed_image = img
        self.showing_original = False
        self.display_image(img)


if __name__ == "__main__":
    root = Tk()
    app = FaceBlurringApp(root)
    root.mainloop()
