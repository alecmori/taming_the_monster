# -*- coding: utf-8 -*-
import keras


def train_model(X, Y, weighted_rewards):
    """TODO: Fill this shit out
    """
    _, num_features = X.shape
    model = keras.models.Sequential()
    model.add(
        keras.layers.Dense(
            units=50,
            activation='sigmoid',
            input_shape=(num_features,),
        ),
    )
    model.add(
        keras.layers.Dense(
            units=1,
        ),
    )
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['accuracy'],
    )
    model.fit(x=X, y=weighted_rewards, epochs=250)
    return model


def score_actions(X, model):
    """TODO: Fill this shit out
    """
    return model.predict(x=X)
