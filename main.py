import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
from model.network import build_model
from data.loader import load_mnist
from training.trainer import train_model, evaluate_model
import tensorflow as tf



MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Config laden
with open("config.json") as f:
    config = json.load(f)

# Daten laden
(x_train, y_train), (x_test, y_test) = load_mnist()

# Modell bauen und kompilieren
model = build_model(config["model"])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training
history = train_model(
    model,
    x_train, y_train,
    epochs=config["training"]["epochs"],
    validation_split=config["training"]["validation_split"],
    batch_size=config["training"]["batch_size"]
)

# Modell speichern
model_path = os.path.join(MODEL_DIR, "mnist_model.keras")
model.save(model_path)
print(f"Model saved to {model_path}")

# Evaluierung
test_loss, test_acc = evaluate_model(model, x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")