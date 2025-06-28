import os
import tensorflow as tf

# Basisverzeichnis ermitteln (eine Ebene höher als dieser loader.py)
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_mnist():
    print("Lade MNIST-Daten aus:", DATA_DIR)
    path = os.path.join(DATA_DIR, "mnist.npz")

    if os.path.exists(path):
        # Hinweis: tf.keras.datasets.mnist.load_data() unterstützt keinen direkten Pfad zu lokalem npz.
        # Man müsste hier das npz manuell laden, falls gewünscht.
        print("Lokale Datei gefunden, direkte Nutzung nicht möglich, lade stattdessen über Keras-API.")
    # Lade den Datensatz über Keras API (cached automatisch in ~/.keras/datasets)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalisieren auf 0–1 Bereich
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    return (x_train, y_train), (x_test, y_test)