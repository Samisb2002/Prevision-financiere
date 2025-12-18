from tensorflow.keras.callbacks import EarlyStopping

def train(model, X_train, y_train, X_val, y_val,
          epochs=50, batch_size=32):

    early_stop = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )
    return history
