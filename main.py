import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
import tensorflow as tf

from model.network import build_model
from data.loader import load_mnist
from training.trainer import train_model, evaluate_model
from recuria.runner import run_recuria_on_model

with open("data/primes_to_1M.txt") as f:
    PRECOMPUTED_PRIMES = set(map(int, f.read().split()))

MODEL_DIR = "./models"
DATA_DIR = "./data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)



with open("config.json") as f:
    config = json.load(f)

(x_train, y_train), (x_test, y_test) = load_mnist()

model = build_model(config["model"])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = train_model(
    model,
    x_train, y_train,
    epochs=config["training"]["epochs"],
    validation_split=config["training"]["validation_split"],
    batch_size=config["training"]["batch_size"]
)

model_path = os.path.join(MODEL_DIR, "mnist_model.keras")
model.save(model_path)
print(f"Model saved to {model_path}")

test_loss, test_acc = evaluate_model(model, x_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")

# Recuria laufen lassen, Output im Data-Ordner
output_path = os.path.join(DATA_DIR, "recuria_results.txt")
run_recuria_on_model(max_steps=1000, output_path=output_path)