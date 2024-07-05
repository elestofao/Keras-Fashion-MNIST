# Elena Lorite Acosta
import numpy as np
import matplotlib.pyplot as plt
from time import time

# Para eliminar los mensajes de error de Cuda sin configurar
import logging, os 
logging.disable(logging.WARNING) 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

# Para las redes
from tensorflow import keras
from keras import layers
from sklearn.metrics import confusion_matrix, classification_report


# Parámetros
num_classes = 10    # Hay 10 clases
input_shape = (28, 28, 1)   # Las imagenes son arrays Numpy 28x28 (en escala de grises)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
x_train, y_train, x_test, y_test = None, None, None, None

def cargarFashionMNIST():
    global x_train, y_train, x_test, y_test
    # Cargar los datos y dividirlos en entrenamiento y test
    # x -> images      y -> labels
    # El conjunto train son los datos de entrenamiento de la red (60,000 ejemplos)
    # El conjunto test son los datos de prueba para poner a prueba a la red (10,000 ejemplos)
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    '''############### PREPROCESAR LOS DATOS ###############'''
    # Escalar imagenes al rango [0, 1]
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    # Dimensiones (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    '''############### VISUALIZAR LOS DATOS ###############'''
    # Visualizamos los datos para comprobar que se han preparado bien
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[y_train[i]])
    plt.show()

    print("x_train shape:", x_train.shape)
    print("x_test shape:", x_test.shape)
    print(x_train.shape[0], "train samples") 
    print(x_test.shape[0], "test samples")   

    # convertir vectores binarios a matrices binarias
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)


def entrenarModelo(model,lote,epocas):
    batch_size = lote           # Cantidad de datos en cada epoca - nº de instancias que se le enseñan al modelo antes de que actualice los pesos
    epochs = epocas             # Epocas - nº de veces que el modelo se expone a la dataset de entrenamiento
    validation_split = 0.1      # Indica la fraccion de los datos de entrenamiento para validacion que se usará para evaluar en cada epoca

    # Compilar el modelo
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # loss = Qué tan exacto es el modelo durante el entrenamiento. Lo que busca es minimizar la funcion para dirigir el modelo en la direccion adecuada
    # 'categorical_crossentropy' es una funcion logaritmica para multi-clases
    # optimizer = Cómo se actualiza el modelo basado en el set de datos y la funcion de perdida
    # metrics = Para monitorear los pasos de entrenamiento y prueba (accuracy es la unica disponible)

    print("\nModelo compilado con éxito!!\n")

    initTime = time()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=validation_split)
    endTime = time()
    executionTime = endTime - initTime
    # devuelve el historial con detalles y metricas calculadas por el modelo en cada epoca y es útil para graficar
    print("\nModelo entrenado con éxito!!\n")

    texto = "Tiempo de entrenamiento: {:.3f} segundos"
    print(texto.format(executionTime))

    evaluarModelo(model,history)
     

def evaluarModelo(model,history):
    # Tasas de perdida y acierto finales del conjunto de entrenamiento
    loss = history.history["loss"][-1]
    accuracy = history.history["accuracy"][-1] 
    print("\nTrain loss:", loss)
    print("Train accuracy:", accuracy)

    # Graficar las tasas para conjunto de entrenamiento y de validacion
    val_loss = history.history["val_loss"]          # tasa de perdida del conjunto de validacion
    val_accuracy = history.history["val_accuracy"]  # tasa de acierto del conjunto de validacion
    train_loss = history.history["loss"]            # tasa de perdida del conjunto de entrenamiento
    train_accuracy = history.history["accuracy"]    # tasa de acierto del conjunto de entrenamiento
    epochs = range(len(history.history["loss"]))

    print("\nValidation loss:", val_loss[-1])
    print("Validation accuracy:", val_accuracy[-1])

    plt.figure(figsize=(15,5))
    #plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, label = "training_loss")
    plt.plot(epochs, val_loss, label = "val_loss")
    plt.title("Loss")
    plt.xlabel("epochs")
    plt.legend()
    #plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, label = "training_accuracy")
    plt.plot(epochs, val_accuracy, label = "val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("epochs")
    plt.legend()
    
    # Tasa de acierto y perdida para el conjunto test
    print('\n')
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print("\nTest loss:", test_loss)
    print("Test accuracy:", test_acc)

    # Matriz de confusion con conjunto de test
    y_pred = model.predict(x_test)
    y_pred_index = np.argmax(y_pred,axis=1)
    y_test_index = np.argmax(y_test,axis=1)

    cm = confusion_matrix(y_test_index, y_pred_index)
    print('\nConfusion matrix', cm)

    plt.figure(figsize=(8,6))
    plot_matrizConfusion(cm)
    cr= classification_report(y_test_index, y_pred_index, target_names=class_names)
    print(cr)   

    plt.show()
    

def plot_matrizConfusion(cm):
    title = 'Confusion Matrix'
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')


def Keras_MLP1_Fashion(lote,epocas):
    cargarFashionMNIST()

    # se crea el modelo
    model = keras.Sequential([
        keras.Input(shape=input_shape),     # En la primera capa siempre se especifica el tamaño del input
        layers.Flatten(),                   # Aplana los datos a un array unidimensional 28x28 = 784
        # Capa oculta Dense 
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax"),    # Capa de salida de 10 nodos softmax
    ])

    model.summary()

    entrenarModelo(model,lote,epocas)


def Keras_MLP2_Fashion(lote,epocas):
    cargarFashionMNIST()

    # se crea el modelo
    model = keras.Sequential([
        keras.Input(shape=input_shape),     # Capa de entrada
        layers.Flatten(),                   # Aplana los datos
        # Capa oculta Dense 1
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.25),
        # Capa oculta Dense 2
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax"),    # Capa de salida
    ])

    model.summary()

    entrenarModelo(model,lote,epocas)


def Keras_CNN_Fashion(lote,epocas):
    cargarFashionMNIST()

    # se crea el modelo
    model = keras.Sequential([
        keras.Input(shape=input_shape),     # Capa de entrada
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.2),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),    # Capa de salida
    ])

    model.summary()

    entrenarModelo(model,lote,epocas)

               

if __name__ == '__main__': 

    #### MLP sencillo ####

    print("Resultados MLP sencillo con [batch_size = 1024, val_split = 0.15, epochs = 25]")
    Keras_MLP1_Fashion(1024,25)


    #### MLP mejorado ####

    #print("Resultados MLP mejorado con [batch_size = 1024, val_split = 0.15, epochs = 25]")
    #Keras_MLP2_Fashion(1024,25)


    #### CNN ####

    #print("Resultados CNN con [batch_size = 1024, val_split = 0.15, epochs = 30]")
    #Keras_CNN_Fashion(1024,30)