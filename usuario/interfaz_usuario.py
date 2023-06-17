import tkinter
from PIL import ImageTk, Image
import os
import cv2
import numpy as np
import pandas as pd
import face_recognition
from paquetes.funciones import codificar_datos, tomar_asistencia_entrada, tomar_asistencia_salida


"""Funciones"""


def reconocer_rostros_entrada():
    """Compara rostros del banco de datos con la camara"""
    # --> Ubicacion de imagenes
    ruta_imagenes = r'../administrador/base_datos'

    # --> Variables
    imagenes = []
    nombre_clases = []
    lista_imagenes = os.listdir(ruta_imagenes)

    # --> Procesar imagenes
    for imagen in lista_imagenes:
        imagen_leida = cv2.imread(f'{ruta_imagenes}/{imagen}')
        imagenes.append(imagen_leida)
        nombre_clases.append(os.path.splitext(imagen)[0])

    lista_codificada = codificar_datos(imagenes)

    camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        success, img = camara.read()
        # --> Preparar imagen de camara
        rostro_camara = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        rostro_camara = cv2.cvtColor(rostro_camara, cv2.COLOR_BGR2RGB)

        # --> Encontrar y codificar rostro de la camara
        ubicacion_rostro = face_recognition.face_locations(rostro_camara)
        codificar_rostro = face_recognition.face_encodings(rostro_camara, ubicacion_rostro)

        # --> Comparar rostro con imagenes
        for rostro_codificado, rostro_ubicado in zip(codificar_rostro, ubicacion_rostro):
            coincidencia = face_recognition.compare_faces(lista_codificada, rostro_codificado)
            distancia_rostros = face_recognition.face_distance(lista_codificada, rostro_codificado)
            # print(distancia_rostros)

            # --> Encontrar menor distancia
            indice_coincide = np.argmin(distancia_rostros)
            if coincidencia[indice_coincide]:
                color = (125, 220, 0)
                nombre = nombre_clases[indice_coincide].title()
                tomar_asistencia_entrada(nombre)
            else:
                nombre = "Desconocido"
                color = (50, 50, 255)

            # --> Medidas de rostro
            y1, x2, y2, x1 = rostro_ubicado
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # --> Cubrir rostros
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(
                img,
                nombre,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2
            )
        cv2.imshow("Asistencia de entrada", img)

        # --> Detener ejecucion con escape
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # --> Libera recurso de software
    camara.release()
    # --> Se destruyen las ventanas que se crearon
    cv2.destroyAllWindows()


def reconocer_rostros_salida():
    # --> Revisar asistencia de entrada
    registro_asistencia = pd.read_csv(r"../administrador/asistencia/asistencia.csv")
    nombres_entrada = registro_asistencia.iloc[:, 0].values

    # --> Ubicacion de imagenes
    ruta_imagenes = r'../administrador/base_datos'

    # --> Variables
    imagenes = []
    nombre_clases = []
    lista_imagenes = os.listdir(ruta_imagenes)

    # --> Procesar imagenes
    for imagen in lista_imagenes:
        imagen_leida = cv2.imread(f'{ruta_imagenes}/{imagen}')
        imagenes.append(imagen_leida)
        nombre_clases.append(os.path.splitext(imagen)[0])

    lista_codificada = codificar_datos(imagenes)

    camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        success, img = camara.read()
        # --> Preparar imagen de camara
        rostro_camara = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        rostro_camara = cv2.cvtColor(rostro_camara, cv2.COLOR_BGR2RGB)

        # --> Encontrar y codificar rostro de la camara
        ubicacion_rostro = face_recognition.face_locations(rostro_camara)
        codificar_rostro = face_recognition.face_encodings(rostro_camara, ubicacion_rostro)

        # --> Comparar rostro con imagenes
        for rostro_codificado, rostro_ubicado in zip(codificar_rostro, ubicacion_rostro):
            coincidencia = face_recognition.compare_faces(lista_codificada, rostro_codificado)
            distancia_rostros = face_recognition.face_distance(lista_codificada, rostro_codificado)
            # print(distancia_rostros)

            # --> Encontrar menor distancia
            indice_coincide = np.argmin(distancia_rostros)
            if coincidencia[indice_coincide]:
                # --> Si usuario hizo 'CheckIn'
                if nombre_clases[indice_coincide].title() in nombres_entrada:
                    color = (125, 220, 0)
                    nombre = nombre_clases[indice_coincide].title()
                    tomar_asistencia_salida(nombre)
                else:
                    color = (255, 0, 0)
                    nombre = "Haz checkint"
            else:
                nombre = "Desconocido"
                color = (50, 50, 255)

            # --> Medidas de rostro
            y1, x2, y2, x1 = rostro_ubicado
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

            # --> Cubrir rostros
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), color, cv2.FILLED)
            cv2.putText(
                img,
                nombre,
                (x1 + 6, y2 - 6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2
            )
        cv2.imshow("Asistencia de salida", img)

        # --> Detener ejecucion con escape
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # --> Libera recurso de software
    camara.release()
    # --> Se destruyen las ventanas que se crearon
    cv2.destroyAllWindows()


# --> Inicio de interfaz

"""Configuracion de ventana"""
ventana = tkinter.Tk()
ventana.geometry("500x600")
ventana.title("Check n' work - Usuario")
ventana.iconbitmap("../icono/logo_definitivo.ico")
ventana.config(bg="white")

"""Contenido de ventana"""

# --> Titulos
nombre_ventana = tkinter.Label(text="Usuario", bg="white", font="bold")

# --> Crear botones
boton_asistencia_entrada = tkinter.Button(
    ventana, text="Asistencia entrada", fg="white", bg="green", font="bold", command=reconocer_rostros_entrada)

boton_asistencia_salida = tkinter.Button(
    ventana, text="Asistencia salida", fg="white", bg="green", font="bold", command=reconocer_rostros_salida)

boton_salir = tkinter.Button(ventana, text="Salir", fg="white", bg="red", font="bold", command=ventana.quit)

# --> Ubicar titulos en interfaz
nombre_ventana.place(relx=0.25, rely=0.48, relwidth=0.50, relheight=0.05)

# --> Ubicar botones en interfaz
boton_asistencia_entrada.place(relx=0.10, rely=0.63, relwidth=0.35, relheight=0.08)
boton_asistencia_salida.place(relx=0.55, rely=0.63, relwidth=0.35, relheight=0.08)
#
boton_salir.place(relx=0.275, rely=0.81, relwidth=0.45, relheight=0.08)

# --> Icono en pantalla
imagen_logo = ImageTk.PhotoImage(Image.open("../icono/logo_definitivo.ico"))
label_imagen = tkinter.Label(image=imagen_logo)
label_imagen.place(relx=0.25, rely=0.11, relwidth=0.50, relheight=0.32)

"""Ejecutar ventana"""
ventana.mainloop()
