# -*- coding: utf-8 -*-
import numpy
import staticconf

from taming_the_monster.train import contextual_bandit_utils
from taming_the_monster.train import data_iterator
from taming_the_monster.train import inverse_propensity_weighting
from taming_the_monster.train import model_utils

CONFIG = 'default-config.yaml'


def train_contextual_bandit(iterate_data, train_model, score_actions):
    """
    :param iterate_data:
    :type iterate_data: function
    :param train_model:
    :type train_model: function
    :param score_actions:
    :type score_actions: function

    :returns: The contextual bandit, which is a list of the models and
        probablities you choose them with.
    :rtype: list
    """
    staticconf.YamlConfiguration(CONFIG)
    contextual_bandit = []
    for epoch, training_data in enumerate(iterate_data()):
        _validate_training_data(training_data=training_data, epoch=epoch)
        propensity_info = inverse_propensity_weighting.get_propensity_info(
            possible_actions=training_data['possible_actions'],
            chosen_actions=training_data['chosen_actions'],
            contextual_bandit=contextual_bandit,
            epoch=epoch,
            score_actions=score_actions,
            rewards=training_data['rewards'],
        )
        model = train_model(
            X=numpy.array(training_data['chosen_actions']),
            Y=numpy.array(training_data['rewards']),
            weighted_rewards=numpy.array(
                propensity_info['weighted_rewards'],
            ),
        )
        contextual_bandit = contextual_bandit_utils.add_model(
            model=model,
            contextual_bandit=contextual_bandit,
            possible_actions=training_data['possible_actions'],
            chosen_actions=training_data['chosen_actions'],
            weighted_rewards=propensity_info['weighted_rewards'],
            min_probs=propensity_info['min_probs'],
            score_actions=score_actions,
        )
    return contextual_bandit


def _validate_training_data(training_data, epoch):
    """TODO"""
    if set(
        training_data.keys(),
    ) != {'possible_actions', 'chosen_actions', 'rewards'}:
        raise ValueError(
            'On epoch {epoch}, data did not have proper keys (expected '
            '"possible_actions" (all distinct choices we could have chosen), '
            '"chosen_actions" (the actual choice observed), and "rewards" '
            '(the benefit associated with the chosen action).'
            '\n'
            '"possible_actions" should be a list of scorable feature arrays, '
            '"chosen_actions" should be a scorable feature array, '
            '"rewards" should be a vector of values.'
            '\n'
            'You might want to modify your `iterate_data` function.',
        )


if __name__ == '__main__':
    train_contextual_bandit(
        iterate_data=data_iterator.iterate_data,
        train_model=model_utils.train_model,
        score_actions=model_utils.score_actions,
    )
