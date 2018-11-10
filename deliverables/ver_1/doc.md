# Correo

gomezportillo@correo.ugr.es

# Topología

Actualmente uso un modelo secuencial con tres capas.
* La primera capa transforma  las imágenes de una array 2D de 28x28 píxeles a una array 1D de  784 píxeles (28*28).
* La segunda capa tiene 128 nodos
* La tercera capa tiene 10 nodos softmax

Cada nodo contiene la probabilidad de que la imagen actual pertenezca a cada una de las 10 clases (del 0 al 9).

# Algoritmo de entrenamiento

Para entrear el modelo he usado la funcion fit con 10 épocas.

model.fit(train_images, train_labels, epochs=10)

He comprobado que si aumento las épocas, por ejemplo a 20, obtengo mejroes resultados en el conjunto de entrenamiento pero peores en el conjunto de prueba (~3% peores), lo que achaco al sobreaprendizaje del algoritmo.

# Etiquetas asignadas al conjunto de pruebas

assigned_labels_test.txt

# Tasa de error sobre el conjunto de prueba (%)

7.118523045459005

# Tasa de error sobre el conjunto de entrenamiento (%)

1.0095688432695653

# Tiempo de entrenamiento (en segundos)

24.434173822402954

# Implementación utilizada en la realización del experimento

He usado el lenguaje de programación Python3 con las bibliotecas Keras + Tensorflow como backend.
