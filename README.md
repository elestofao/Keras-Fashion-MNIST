# Keras-Fashion-MNIST

MLP y CNN con Keras para identificar prendas de ropa.

## Introducción

Estas instrucciones proporcionan una copia del proyecto en funcionamiento para fines de desarrollo y pruebas.

### Instalación

**Paso 1. Instalar [Keras](https://github.com/keras-team/keras) y [TensorFlow](https://github.com/tensorflow/tensorflow).**

    pip install tensorflow
    pip install keras==3.1.1

**Paso 2. Clonar el repositorio.**

    git clone https://github.com/elestofao/Keras-Fashion-MNIST.git

## Sobre el código

Se han creado tres clasificadores con el objetivo de comparar resultados y funcionamiento de ellos:

- MLP sencillo con 1 capa oculta.
- MLP mejorado con 2 capas ocultas.
- CNN con dos capas de convolución y una de pooling.

Para cada clasificador hay una función, por lo que en el programa principal se debe descomentar el que se quiera probar y comentar el resto.

## Ejecución

Para ejecutar el/los clasificadores:

    python Keras_FashionMNIST.py

## Resultados
### Acierto

| Clasificador | Test Accuracy |
| --- | --- |
| MLP sencillo | 88.68% |
| MLP mejorado | 89.34% |
| CNN | 92.08% |

### Evolución de las tasas de acierto y pérdida
**MLP sencillo**

![historySimpleMLP](https://github.com/elestofao/Keras-Fashion-MNIST/assets/149679202/35586e11-ec78-4eba-b99a-51e40b5633e3)


**MLP mejorado**

![historyMLP](https://github.com/elestofao/Keras-Fashion-MNIST/assets/149679202/b92d6551-cf54-4c2a-abe1-97b9663c42d4)


**CNN**

![historyCNN](https://github.com/elestofao/Keras-Fashion-MNIST/assets/149679202/5b5f8320-768b-4eb2-8d8a-ef9bdf90e353)


