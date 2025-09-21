import cv2
import face_recognition

# --> Preparar imagenes
img_uriel = face_recognition.load_image_file('images/uriel_prueba.jpg')
img_uriel = cv2.cvtColor(img_uriel, cv2.COLOR_BGR2RGB)


img_test = face_recognition.load_image_file('images/uriel_lentes.jpg')
img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)

# --> Encontrar y codificar rostro
ubicacion_rostro = face_recognition.face_locations(img_uriel)[0]
codificar_rostro = face_recognition.face_encodings(img_uriel)[0]

ubicacion_rostro_test = face_recognition.face_locations(img_test)[0]
codificar_rostro_test = face_recognition.face_encodings(img_test)[0]

# --> Ubicacion de rostro
# Imagen 1
cv2.rectangle(
    img_uriel,
    (ubicacion_rostro[3], ubicacion_rostro[0]),
    (ubicacion_rostro[1], ubicacion_rostro[2]),
    (255, 0, 255), 2
)
# Imagen 2
cv2.rectangle(
    img_test,
    (ubicacion_rostro_test[3], ubicacion_rostro_test[0]),
    (ubicacion_rostro_test[1], ubicacion_rostro_test[2]),
    (255, 0, 255), 2
)

# --> Comparar rostros por codificacion
# Verdadero o Falso
hay_coincidencia = face_recognition.compare_faces(
    [codificar_rostro], codificar_rostro_test
)
# Mientras menor sea el numero mas coincidencia habra
distancia_rostros = face_recognition.face_distance(
    [codificar_rostro], codificar_rostro_test
)
print(hay_coincidencia, distancia_rostros)
cv2.putText(
    img_test,
    f'{hay_coincidencia} {round(distancia_rostros[0],2)}',
    (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2
)

# --> Ver imagen en formato original
cv2.imshow("uriel martinez", img_uriel)
cv2.imshow("uriel test", img_test)
cv2.waitKey(0)
cv2.destroyAllWindows()
