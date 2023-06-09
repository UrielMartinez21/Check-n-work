import os
import cv2
import pandas as pd
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


def tomar_asistencia_entrada(nombre):
    """Registro de asistencia usuarios para entrar"""

    with open("../administrador/asistencia/asistencia.csv", "r+") as f:
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


def tomar_asistencia_salida(nombre):
    """Registro de asistencia usuarios para entrar"""
    # --> Validar que no haya checkout
    registro_asistencia = pd.read_csv(r"../administrador/asistencia/asistencia.csv")
    registro = registro_asistencia[registro_asistencia["Nombre"] == nombre].values
    registro_salida = str(registro[0][2])

    # --> Si el usuario no ha hecho 'CheckOut'
    if registro_salida == "nan":
        # --> Crear fecha
        fecha = datetime.now()
        fecha_string = fecha.strftime("%H:%M:%S")

        # --> Sobreescribir fila del registro encontrado
        registro_asistencia.loc[registro_asistencia["Nombre"] == nombre, "Salida"] = fecha_string

        # --> Obtener valores de DataFrame
        datos = registro_asistencia.values

        # --> Sobreescribir archivo csv
        with open("../administrador/asistencia/asistencia.csv", "r+") as f:
            f.write(f"Nombre,Entrada,Salida")
            for dato in datos:
                f.writelines(f"\n{dato[0]},{dato[1]},{dato[2]}")
