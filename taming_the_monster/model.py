# -*- coding: utf-8 -*-
import keras


def train_single_model(X, Y):
    """TODO DESCRIBE THIS
    """
    _, num_features = X.shape
    model = keras.models.Sequential()
    model.add(keras.layers.Dense(units=100, input_shape=(num_features,)))
    model.add(keras.layers.Dense(units=20, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    model.fit(x=X, y=Y, epochs=5)
    return model


def train_contextual_bandit_model(X, Y, potential_actions, model_ensemble):
    """TODO DESCRIBE THIS
    """
    raise NotImplementedError


def _get_inverse_propensity_weight(X, potential_actions, model_ensemble):
    """TODO DESCRIBE THIS
    """
    raise NotImplementedError


def _get_probability_of_action(actions, model_ensemble):
    """TODO DESCRIBE THIS
    """
    raise NotImplementedError
