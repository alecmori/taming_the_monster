# -*- coding: utf-8 -*-
import keras


def train_model(X, Y, weights):
    """TODO: Fill this shit out
    """
    _, num_features = X.shape
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=20, input_shape=(num_features,)))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(x=X, y=Y, sample_weight=weights, epochs=10)
    return model


def score_actions(X, model):
    """TODO: Fill this shit out
    """
    return model.predict(x=X)
