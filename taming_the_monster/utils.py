# -*- coding: utf-8 -*-
def find_scoring_function(model):
    """Attempts to find scoring function

    :param model: The model you are using in the contextual bandit
    :type model: any(keras.models.Model)
    """
    raise NotImplementedError(
        'Error! Do not have built in scoring for this function. Please pass '
        'in scoring function name in config.',
    )
