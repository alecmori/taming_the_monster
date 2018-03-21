# -*- coding: utf-8 -*-
import staticconf

from taming_the_monster import contextual_bandit_utils
from taming_the_monster import data_iterator
from taming_the_monster import inverse_propensity_weighting
from taming_the_monster import model

CONFIG = 'default-config.yaml'


def main():
    staticconf.YamlConfiguration(CONFIG)
    contextual_bandit = []
    for training_data in data_iterator.iterate_data():
        new_model = model.train_model(
            X=training_data['chosen_actions'],
            Y=training_data['rewards'],
            weights=inverse_propensity_weighting(
                possible_actions=training_data['possible_actions'],
                chosen_actions=training_data['chosen_actions'],
                contextual_bandit=contextual_bandit,
            ),
        )
        contextual_bandit = contextual_bandit_utils.add_model(
            model=new_model,
            contextual_bandit=contextual_bandit,
        )


if __name__ == '__main__':
    main()
