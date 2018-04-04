# -*- coding: utf-8 -*-
import keras


def train_single_model(X, Y):
    """TODO DESCRIBE THIS
    """
    _, num_features = X.shape
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=1,
            activation='sigmoid',
            input_shape=(num_features,),
        ),
    )
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(x=X, y=Y, epochs=5)
    return model
