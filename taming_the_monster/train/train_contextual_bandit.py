# -*- coding: utf-8 -*-
import staticconf

from taming_the_monster import contextual_bandit_utils
from taming_the_monster import data_iterator
from taming_the_monster import inverse_propensity_weighting
from taming_the_monster import model_utils

CONFIG = 'default-config.yaml'


def main():
    staticconf.YamlConfiguration(CONFIG)
    contextual_bandit = []
    for epoch, training_data in enumerate(data_iterator.iterate_data()):
        propensity_info = inverse_propensity_weighting.get_propensity_info(
            possible_actions=training_data['possible_actions'],
            chosen_actions=training_data['chosen_actions'],
            contextual_bandit=contextual_bandit,
            epoch=epoch,
        )
        model = model_utils.train_model(
            X=training_data['chosen_actions'],
            Y=training_data['rewards'],
            weights=propensity_info['weights'],
        )
        contextual_bandit = contextual_bandit_utils.add_model(
            model=model,
            contextual_bandit=contextual_bandit,
            possible_actions=training_data['possible_actions'],
            chosen_actions=training_data['chosen_actions'],
            Y=training_data['rewards'],
            weights=propensity_info['weights'],
            min_probs=propensity_info['min_probs'],
        )
    # TODO: Save contextual bandit here


if __name__ == '__main__':
    main()
