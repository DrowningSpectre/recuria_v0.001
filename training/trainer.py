def train_model(model, x_train, y_train, epochs, validation_split, batch_size):
    return model.fit(
        x_train, y_train,
        epochs=epochs,
        validation_split=validation_split,
        batch_size=batch_size
    )

def evaluate_model(model, x_test, y_test):
    return model.evaluate(x_test, y_test)