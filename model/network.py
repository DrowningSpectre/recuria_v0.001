from tensorflow.keras import layers, models # type: ignore

def build_model(config):
    model = models.Sequential([
    layers.Input(shape=tuple(config["input_shape"])),
    layers.Flatten(),
    layers.Dense(config["hidden_units"], activation='relu'),
    layers.Dropout(config["dropout"]),
    layers.Dense(config["output_classes"], activation='softmax')
])
    return model

