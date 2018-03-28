# -*- coding: utf-8 -*-
import numpy
import staticconf

from taming_the_monster.train import contextual_bandit_utils
from taming_the_monster.train import data_iterator
from taming_the_monster.train import inverse_propensity_weighting
from taming_the_monster.train import model_utils

CONFIG = 'default-config.yaml'


def train_contextual_bandit(iterate_data, train_model, score_actions):
    staticconf.YamlConfiguration(CONFIG)
    contextual_bandit = []
    for epoch, training_data in enumerate(iterate_data()):
        propensity_info = inverse_propensity_weighting.get_propensity_info(
            possible_actions=training_data['possible_actions'],
            chosen_actions=training_data['chosen_actions'],
            contextual_bandit=contextual_bandit,
            epoch=epoch,
            score_actions=score_actions,
        )
        model = train_model(
            X=numpy.array(training_data['chosen_actions']),
            Y=numpy.array(training_data['rewards']),
            weights=numpy.array(propensity_info['weights']),
        )
        contextual_bandit = contextual_bandit_utils.add_model(
            model=model,
            contextual_bandit=contextual_bandit,
            possible_actions=training_data['possible_actions'],
            chosen_actions=training_data['chosen_actions'],
            Y=training_data['rewards'],
            weights=propensity_info['weights'],
            min_probs=propensity_info['min_probs'],
            score_actions=score_actions,
        )
    return contextual_bandit


if __name__ == '__main__':
    train_contextual_bandit(
        iterate_data=data_iterator.iterate_data,
        train_model=model_utils.train_model,
        score_actions=model_utils.score_actions,
    )
