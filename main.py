import tkinter
from PIL import ImageTk,Image
import os
import cv2
import numpy as np
import face_recognition
from paquetes.funciones import crear_carpeta,codificar_datos,tomar_asistencia

"""Funciones"""
def extraer_rostros():
    # --> Ubicacion de imagenes
    ruta_entrada = "entrada_imagenes"

    #--> Carpeta sino existe
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
            cv2.rectangle(ver_imagen, (x, y), (x+w, y+h), (0, 255, 0), 2)
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

def reconocer_rostros():
    # --> Ubicacion de imagenes
    ruta_imagenes = 'base_datos'

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
                tomar_asistencia(nombre)
            else:
                nombre = "Desconocido"
                color = (50, 50, 255)

            # --> Medidas de rostro
            y1, x2, y2, x1 = rostro_ubicado
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            # --> Cubrir rostros
            cv2.rectangle(img, (x1, y1), (x2, y2), color , 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), color , cv2.FILLED)
            cv2.putText(
                img,
                nombre,
                (x1+6, y2-6),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2
            )
        cv2.imshow("Webcam", img)

        # --> Detener ejecucion con escape
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    # --> Libera recurso de software
    camara.release()
    # --> Se destruyen las ventanas que se crearon
    cv2.destroyAllWindows()

def revisar_asistencia():
    os.system("start EXCEL.EXE asistencia/asistencia.csv")


"""Configuracion de ventana"""
ventana = tkinter.Tk()
ventana.geometry("295x600")
ventana.title("Check n' work")
ventana.iconbitmap("icono/icono.ico")
ventana.config(bg="white")

"""Contenido de ventana"""
#--> Crear botones
boton_extraer_rostros = tkinter.Button(ventana,text="Extraer rostros",fg="white",bg="green",font="bold",command=extraer_rostros)
boton_tomar_asistencia = tkinter.Button(ventana,text="Tomar asistencia",fg="white",bg="green",font="bold",command=reconocer_rostros)
boton_revisar_asistencia = tkinter.Button(ventana,text="Revisar asistencia",fg="white",bg="green",font="bold",command=revisar_asistencia)
boton_salir = tkinter.Button(ventana,text="Salir",fg="white",bg="red",font="bold",command=ventana.quit)

#--> Ubicar botones
boton_extraer_rostros.place(relx=0.1694,rely=0.4625,relwidth=0.6610,relheight=0.0833)
boton_tomar_asistencia.place(relx=0.1694,rely=0.5791,relwidth=0.6610,relheight=0.0833)
boton_revisar_asistencia.place(relx=0.1694,rely=0.6958,relwidth=0.6610,relheight=0.0833)
boton_salir.place(relx=0.1694,rely=0.8125,relwidth=0.6610,relheight=0.0833)


#--> Icono en pantalla
imagen_logo = ImageTk.PhotoImage(Image.open("icono/icono.ico"))
label_imagen = tkinter.Label(image=imagen_logo)
label_imagen.place(relx=0.1694,rely=0.1041,relwidth=0.6610,relheight=0.325)

"""Ejecutar ventana"""
ventana.mainloop()