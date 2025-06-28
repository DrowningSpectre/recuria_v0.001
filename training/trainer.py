def train_model(model, x_train, y_train, epochs, validation_split, batch_size):
    """
    Trainiert das Modell mit den angegebenen Parametern.
    """
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_split=validation_split,
        batch_size=batch_size
    )
    return history

def evaluate_model(model, x_test, y_test):
    """
    Evaluierung des Modells auf den Testdaten.
    """
    return model.evaluate(x_test, y_test)