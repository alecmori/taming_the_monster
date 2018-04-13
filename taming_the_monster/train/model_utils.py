# -*- coding: utf-8 -*-
from sklearn import linear_model


def train_model(X, Y, weighted_rewards):
    """TODO: Fill this shit out
    """
    model = linear_model.LinearRegression()
    model.fit(X=X, y=weighted_rewards)
    return model


def score_actions(X, model):
    """TODO: Fill this shit out
    """
    return model.predict(X=X)
