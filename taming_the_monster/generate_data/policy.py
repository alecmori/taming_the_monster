# -*- coding: utf-8 -*-
import functools

import numpy


def get_policy(complexity, num_features, mean, std):
    """ Builds a weird policy that can't be captured nonlinearly

    See :func _execute_policy: for exactly how the policy works

    :param complexity: The number of xor layers to use in the policy
    :type complexity: int
    :param num_features: The number of features that we are working
        with
    :type num_features: int
    :param mean: The mean of the coefficients in the final policy
    :type mean: float
    :param std: The standard deviation of the coefficients in the final
        policy
    :type std: float

    :returns: The actual policy - it takes in the feature vector and
        returns a value between 0 and 1
    :rtype: function
    """
    xor_layers = [
        numpy.random.random((1, num_features))
        for _ in range(complexity)
    ]
    final_layer = numpy.random.normal(
        loc=mean,
        scale=std,
        size=(num_features, 1),
    )
    return lambda X: _execute_policy(
        X=X,
        xor_layers=xor_layers,
        final_layer=final_layer,
    )


def _execute_policy(X, xor_layers, final_layer):
    """Executes the policy we are trying to learn

    For each xor_layer, we check if X > xor_layer and XOR them all
        together. After all of this, we will get a feature vector of
        0 and 1's. We then take the dot product with the final layer
        and return the sigmoid.

    :param X: The feature vector we are generating the probability of
        success for
    :type X: numpy.array
    :param xor_layers: All of the non-linear XOR layers that our model
        will try to capture
    :type xor_layers: list of numpy.array
    :param final_layer: The final layer of the policy we are trying to
        learn
    :type: numpy.array

    :returns: Scores for each feature vector
    :rtype: numpy.array
    """
    executed_xor_policy = functools.reduce(
        lambda x, y: x ^ y,
        [
            X > xor_layer
            for xor_layer in xor_layers
        ],
    )
    return 1 / (1 + numpy.exp(-1 * executed_xor_policy.dot(final_layer)))
