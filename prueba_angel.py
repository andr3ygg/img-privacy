import os
import cv2
import numpy as np
from tkinter import filedialog, Tk, simpledialog, messagebox
import face_recognition

faces_dir = "faces_db"
os.makedirs(faces_dir, exist_ok=True)

def registrar_rostro(image, face_locations):
    nombre = simpledialog.askstring("Registrar rostro", "Introduce el nombre para este rostro:")
    if not nombre:
        messagebox.showinfo("Registro cancelado", "No se introdujo ningún nombre.")
        return None
    # Extraer encoding del primer rostro detectado
    encodings = face_recognition.face_encodings(image, face_locations)
    if not encodings:
        messagebox.showinfo("Error", "No se pudo extraer el rostro.")
        return None
    np.save(os.path.join(faces_dir, f"{nombre}.npy"), encodings[0])
    messagebox.showinfo("Registro exitoso", f"Rostro de '{nombre}' registrado correctamente.")
    return nombre

def cargar_embeddings(nombres):
    embeddings = []
    for nombre in nombres:
        path = os.path.join(faces_dir, f"{nombre}.npy")
        if os.path.isfile(path):
            embeddings.append((nombre, np.load(path)))
        else:
            print(f"Advertencia: No se encontró el rostro registrado para '{nombre}'.")
    return embeddings

def main():
    Tk().withdraw()
    image_path = filedialog.askopenfilename(title='Selecciona una imagen', filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not image_path:
        print("No se seleccionó ninguna imagen.")
        return

    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)

    nombres_registrados = [os.path.splitext(f)[0] for f in os.listdir(faces_dir) if f.endswith('.npy')]

    if not nombres_registrados:
        respuesta = messagebox.askyesno("No hay registros", "No hay rostros registrados. ¿Quieres registrar un rostro de esta imagen?")
        if respuesta:
            nombre = registrar_rostro(rgb_image, face_locations)
            if not nombre:
                return
            nombres_registrados.append(nombre)
        else:
            messagebox.showinfo("Salir", "No se puede continuar sin registros.")
            return

    while True:
        opciones = nombres_registrados + ["[Añadir nuevo]", "[Difuminar todos]", "[Terminar selección]"]
        seleccion = simpledialog.askstring("Difuminar", f"¿A quién quieres difuminar?\nOpciones: {', '.join(opciones)}\n(Escribe el nombre, '[Añadir nuevo]', '[Difuminar todos]' o '[Terminar selección]')")
        if not seleccion or seleccion == "[Terminar selección]":
            return
        if seleccion == "[Añadir nuevo]":
            nombre = registrar_rostro(rgb_image, face_locations)
            if nombre:
                nombres_registrados.append(nombre)
            continue
        if seleccion == "[Difuminar todos]":
            embeddings = None
            break
        if seleccion in nombres_registrados:
            embeddings = cargar_embeddings([seleccion])
            break
        else:
            messagebox.showinfo("No encontrado", f"No se encontró el nombre '{seleccion}'.")
            continue

    # Procesar imagen
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        difuminar = False
        if embeddings is None:
            difuminar = True
        else:
            for nombre, emb in embeddings:
                match = face_recognition.compare_faces([emb], face_encoding, tolerance=0.5)[0]
                if match:
                    difuminar = True
                    break
        if difuminar:
            face = image[top:bottom, left:right]
            if face.size > 0:
                blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
                image[top:bottom, left:right] = blurred_face

    cv2.imshow("Caras con blur", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    name, ext = os.path.splitext(image_path)
    output_path = f"{name}_blurred{ext}"
    cv2.imwrite(output_path, image)
    print(f"Imagen guardada en: {output_path}")

if __name__ == "__main__":
    main()