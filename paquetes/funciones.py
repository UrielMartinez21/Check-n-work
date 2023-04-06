import os
import cv2
import face_recognition
from datetime import datetime


def crear_carpeta(nombre):
    """Crear carpetas dentro del proyecto"""

    if not os.path.exists(nombre):
        os.mkdir(nombre)
        print("[+]Carpeta creada")


def codificar_datos(lista_imagenes):
    """Procesa y codifica el argumento recibido"""

    lista_codificaciones = []
    for img in lista_imagenes:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        codificacion = face_recognition.face_encodings(img)[0]
        lista_codificaciones.append(codificacion)
    return lista_codificaciones


def tomar_asistencia(nombre):
    """Registro de asistencia usuarios no revisados"""

    with open("asistencia/asistencia.csv", "r+") as f:
        lista_informacion = f.readlines()
        lista_nombres = []
        # --> Nombres registrados en documento
        for linea in lista_informacion:
            entrada = linea.split(",")
            lista_nombres.append(entrada[0])
        # --> Validar si el nombre no tomo asistencia
        if nombre not in lista_nombres:
            fecha = datetime.now()
            fecha_string = fecha.strftime("%H:%M:%S")
            f.writelines(f"\n{nombre},{fecha_string}")
