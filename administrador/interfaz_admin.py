import tkinter
from tkinter import filedialog
from PIL import ImageTk, Image
import os
import cv2
from paquetes.funciones import crear_carpeta

"""Funciones"""


def extraer_rostros():
    """Extraer rostros de un banco de datos"""

    # --> Ubicacion de imagenes
    ruta_entrada = "entrada_imagenes"

    # --> Carpeta sino existe
    crear_carpeta("base_datos")

    # --> Detector facial
    clasificador_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # --> Nombre de los rostros
    count = 0
    # --> Leer imagenes
    print("[+]Imagenes procesadas:")
    for nombre_imagen in os.listdir(ruta_entrada):
        print("\t-->", nombre_imagen)
        # -->Ruta de imagen
        imagen = cv2.imread(ruta_entrada + "/" + nombre_imagen)
        ver_imagen = cv2.imread(ruta_entrada + "/" + nombre_imagen)

        # --> Deteccion de rostros
        rostros = clasificador_rostros.detectMultiScale(imagen, 1.1, 5)
        for (x, y, w, h) in rostros:
            # --> Comprobar deteccion de rostro
            cv2.rectangle(ver_imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # --> Rostro en base en ancho y alto
            rostro = imagen[y:y + h, x:x + w]
            # --> Redimencionar imagen a 200 px
            rostro = cv2.resize(rostro, (200, 200))

            # --> Guardar imagen en 'base_datos'
            cv2.imwrite("base_datos/" + str(count) + ".jpg", rostro)
            count += 1

            # --> Ver imagenes
            cv2.imshow("Rostro", ver_imagen)
            cv2.waitKey(0)
    print("[+]Proceso terminado")

    # --> Destruir ventanas creadas
    cv2.destroyAllWindows()


def reconocimiento_docente():
    """Reconocimiento individual de cada persona"""

    # --> Definir formatos compatibles
    ruta_imagen = filedialog.askopenfilename(filetypes=[
        ("image", ".jpeg"),
        ("image", ".png"),
        ("image", ".jpg")])

    # print(ruta_imagen)
    nombre_imagen = ruta_imagen.split("/")[-1].split(".")[0]
    # print(nombre_imagen)

    # --> Carpeta sino existe
    crear_carpeta("base_datos")

    # --> Detector facial
    clasificador_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # print("\t-->", nombre_imagen)
    # -->Ruta de imagen
    imagen = cv2.imread(ruta_imagen)
    ver_imagen = cv2.imread(ruta_imagen)

    # --> Deteccion de rostros
    rostros = clasificador_rostros.detectMultiScale(imagen, 1.1, 5)
    for (x, y, w, h) in rostros:
        # --> Comprobar deteccion de rostro
        cv2.rectangle(ver_imagen, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # --> Rostro en base en ancho y alto
        rostro = imagen[y:y + h, x:x + w]
        # --> Redimencionar imagen a 200 px
        rostro = cv2.resize(rostro, (200, 200))

        # --> Guardar imagen en 'base_datos'
        cv2.imwrite("base_datos/" + str(nombre_imagen) + ".jpg", rostro)
        # count += 1

        # --> Ver imagenes
        cv2.imshow("Rostro", ver_imagen)
        cv2.waitKey(0)

    print("[+]Proceso terminado")

    # --> Destruir ventanas creadas
    cv2.destroyAllWindows()


def revisar_asistencia():
    os.system("start EXCEL.EXE asistencia/asistencia.csv")


# --> Inicio de interfaz

"""Configuracion de ventana"""
ventana = tkinter.Tk()
ventana.geometry("500x600")
ventana.title("Check n' work - Administrador")
ventana.iconbitmap("../icono/icono.ico")
ventana.config(bg="white")

"""Contenido de ventana"""

# --> Titulos
nombre_ventana = tkinter.Label(text="Administrador", bg="white", font="bold")

# --> Crear botones
boton_extraer_rostros = tkinter.Button(
    ventana, text="Extraer rostros", fg="white", bg="green", font="bold", command=extraer_rostros)

boton_extraer_uno = tkinter.Button(
    ventana, text="+", fg="white", bg="green", font="bold", command=reconocimiento_docente)

boton_revisar_asistencia = tkinter.Button(
    ventana, text="Revisar asistencia", fg="white", bg="green", font="bold", command=revisar_asistencia)

boton_salir = tkinter.Button(ventana, text="Salir", fg="white", bg="red", font="bold", command=ventana.quit)

# --> Ubicar titulos en interfaz
nombre_ventana.place(relx=0.25, rely=0.47, relwidth=0.50, relheight=0.05)

# --> Ubicar botones en interfaz

boton_extraer_rostros.place(relx=0.275, rely=0.58, relwidth=0.315, relheight=0.08)
boton_extraer_uno.place(relx=0.6125, rely=0.58, relwidth=0.09, relheight=0.08)

boton_revisar_asistencia.place(relx=0.275, rely=0.70, relwidth=0.45, relheight=0.08)

boton_salir.place(relx=0.275, rely=0.82, relwidth=0.45, relheight=0.08)

# --> Icono en pantalla
imagen_logo = ImageTk.PhotoImage(Image.open("../icono/icono.ico"))
label_imagen = tkinter.Label(image=imagen_logo)
label_imagen.place(relx=0.25, rely=0.10, relwidth=0.50, relheight=0.32)

"""Ejecutar ventana"""
ventana.mainloop()
