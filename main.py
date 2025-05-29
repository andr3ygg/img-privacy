import os
import cv2
import numpy as np
from urllib.request import urlretrieve
from tkinter import filedialog, Tk, Button, Label, messagebox, Canvas
from PIL import Image, ImageTk
import pytesseract


class DescargadorModelos:
    @staticmethod
    def descargar(ruta_prototxt, url_prototxt, ruta_modelo, url_modelo):
        os.makedirs(os.path.dirname(ruta_prototxt), exist_ok=True)
        if not os.path.isfile(ruta_prototxt):
            print("Descargando deploy.prototxt...")
            urlretrieve(url_prototxt, ruta_prototxt)
        if not os.path.isfile(ruta_modelo):
            print("Descargando modelo caffemodel...")
            urlretrieve(url_modelo, ruta_modelo)


class DetectorCaras:
    def __init__(self, carpeta_modelos="models"):
        self.url_prototxt = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        self.url_modelo = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        self.ruta_prototxt = os.path.join(carpeta_modelos, "deploy.prototxt")
        self.ruta_modelo = os.path.join(carpeta_modelos, "res10_300x300_ssd_iter_140000.caffemodel")

        DescargadorModelos.descargar(self.ruta_prototxt, self.url_prototxt, self.ruta_modelo, self.url_modelo)
        self.net = cv2.dnn.readNetFromCaffe(self.ruta_prototxt, self.ruta_modelo)

    def detectar_y_desenfocar_caras(self, imagen, umbral_confianza=0.5):
        alto0, ancho0 = imagen.shape[:2]
        imagen_para_red = imagen.copy()
        max_tamano = 2000

        if max(alto0, ancho0) > max_tamano:
            escala = max_tamano / max(alto0, ancho0)
            imagen_para_red = cv2.resize(imagen, (int(ancho0 * escala), int(alto0 * escala)))
        else:
            escala = 1.0

        alto, ancho = imagen_para_red.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(imagen_para_red, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detecciones = self.net.forward()

        for i in range(detecciones.shape[2]):
            confianza = detecciones[0, 0, i, 2]
            if confianza > umbral_confianza:
                caja = detecciones[0, 0, i, 3:7] * np.array([ancho, alto, ancho, alto])
                (x1, y1, x2, y2) = caja.astype("int")
                x1, y1 = int(x1 / escala), int(y1 / escala)
                x2, y2 = int(x2 / escala), int(y2 / escala)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(ancho0, x2), min(alto0, y2)

                cara = imagen[y1:y2, x1:x2]
                if cara.size == 0:
                    continue
                area_cara = (x2 - x1) * (y2 - y1)
                k = max(15, int(np.sqrt(area_cara) // 2) | 1)
                cara_desenfocada = cv2.GaussianBlur(cara, (k, k), 30)
                imagen[y1:y2, x1:x2] = cara_desenfocada

        return imagen


class DetectorTexto:
    def __init__(self):
        # Configuración para pytesseract: motor LSTM y modo página uniforme
        self.config = '--oem 3 --psm 6'

    def detectar_y_desenfocar_texto(self, imagen, palabras_clave=None):
        """
        Detecta texto con pytesseract y desenfoca áreas que contengan texto sensible.
        palabras_clave: lista de palabras que indican texto sensible (ejemplo: 'Calle', 'Tarjeta', etc.)
        """
        if palabras_clave is None:
            palabras_clave = ['Calle', 'Tarjeta', 'ID', 'Número', 'Dirección', 'Crédito', 'Pasaporte']

        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        datos = pytesseract.image_to_data(gris, config=self.config, output_type=pytesseract.Output.DICT)

        n_cajas = len(datos['level'])
        for i in range(n_cajas):
            texto = datos['text'][i].strip()
            if any(palabra.lower() in texto.lower() for palabra in palabras_clave) and texto != '':
                (x, y, w, h) = (datos['left'][i], datos['top'][i], datos['width'][i], datos['height'][i])
                region = imagen[y:y+h, x:x+w]
                if region.size > 0:
                    k = max(15, (min(w, h) // 2) | 1)
                    imagen[y:y+h, x:x+w] = cv2.GaussianBlur(region, (k, k), 30)

        return imagen


class ProcesadorImagen:
    def __init__(self):
        self.detector_caras = DetectorCaras()
        self.detector_texto = DetectorTexto()

    def procesar(self, imagen, desenfocar_caras=True, desenfocar_texto=True):
        resultado = imagen.copy()
        if desenfocar_caras:
            resultado = self.detector_caras.detectar_y_desenfocar_caras(resultado)
        if desenfocar_texto:
            resultado = self.detector_texto.detectar_y_desenfocar_texto(resultado)
        return resultado


class AplicacionDesenfoque:
    def __init__(self, root):
        self.root = root
        self.root.title("Detector y Desenfoque de Privacidad")

        self.ruta_imagen = None
        self.imagen_original = None
        self.imagen_procesada = None
        self.mostrando_original = True
        self.procesador = ProcesadorImagen()

        self.etiqueta = Label(root, text="Selecciona una imagen")
        self.etiqueta.pack()

        self.canvas = Canvas(root, width=500, height=400)
        self.canvas.pack()

        Button(root, text="Cargar Imagen", command=self.cargar_imagen).pack(pady=2)
        Button(root, text="Aplicar Blur a Caras y Texto", command=self.aplicar_blur).pack(pady=2)
        Button(root, text="Comparar Antes/Después", command=self.alternar_vista).pack(pady=2)
        Button(root, text="Deshacer Blur", command=self.deshacer_blur).pack(pady=2)
        Button(root, text="Guardar Imagen", command=self.guardar_imagen).pack(pady=2)

    def cargar_imagen(self):
        ruta = filedialog.askopenfilename(
            title='Selecciona una imagen',
            filetypes=[("Archivos de imagen", "*.jpg;*.jpeg;*.png")]
        )
        if ruta:
            self.ruta_imagen = ruta
            self.imagen_original = cv2.imread(ruta)
            self.imagen_procesada = None
            self.mostrando_original = True
            self.mostrar_imagen(self.imagen_original)
        else:
            messagebox.showinfo("Info", "No se seleccionó ninguna imagen.")

    def mostrar_imagen(self, imagen_cv):
        rgb = cv2.cvtColor(imagen_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        pil_img.thumbnail((500, 400))
        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(250, 200, image=self.tk_img)

    def aplicar_blur(self):
        if self.imagen_original is not None:
            self.imagen_procesada = self.procesador.procesar(self.imagen_original, desenfocar_caras=True, desenfocar_texto=True)
            self.mostrando_original = False
            self.mostrar_imagen(self.imagen_procesada)
        else:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")

    def alternar_vista(self):
        if self.imagen_original is None:
            messagebox.showwarning("Advertencia", "Primero carga una imagen.")
            return
        if self.imagen_procesada is None:
            messagebox.showinfo("Info", "Primero aplica el desenfoque.")
            return
        self.mostrando_original = not self.mostrando_original
        imagen = self.imagen_original if self.mostrando_original else self.imagen_procesada
        self.mostrar_imagen(imagen)

    def deshacer_blur(self):
        if self.imagen_original is not None:
            self.imagen_procesada = None
            self.mostrando_original = True
            self.mostrar_imagen(self.imagen_original)
            messagebox.showinfo("Info", "Se restauró la imagen original.")
        else:
            messagebox.showwarning("Advertencia", "No hay imagen cargada.")

    def guardar_imagen(self):
        if self.imagen_procesada is not None:
            nombre, ext = os.path.splitext(self.ruta_imagen)
            ruta_salida = f"{nombre}_desenfocada{ext}"
            cv2.imwrite(ruta_salida, self.imagen_procesada)
            messagebox.showinfo("Guardado", f"Imagen guardada en: {ruta_salida}")
        else:
            messagebox.showwarning("Advertencia", "No hay imagen procesada para guardar.")


if __name__ == "__main__":
    root = Tk()
    app = AplicacionDesenfoque(root)
    root.mainloop()
