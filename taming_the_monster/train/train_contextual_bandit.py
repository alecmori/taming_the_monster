# -*- coding: utf-8 -*-
import staticconf

from taming_the_monster import contextual_bandit_utils
from taming_the_monster import data_iterator
from taming_the_monster import filter_data
from taming_the_monster import inverse_propensity_weighting
from taming_the_monster import model

CONFIG = 'default-config.yaml'


def main():
    staticconf.YamlConfiguration(CONFIG)
    contextual_bandit = []
    for training_data in data_iterator.iterate_data():
        filtered_data = filter_data.filter_data(
            training_data=training_data,
            contextual_bandit=contextual_bandit,
        )
        new_model = model.train_model(
            X=filtered_data['chosen_actions'],
            Y=filtered_data['rewards'],
            weights=inverse_propensity_weighting(
                possible_actions=filtered_data['possible_actions'],
                chosen_actions=filtered_data['chosen_actions'],
                contextual_bandit=contextual_bandit,
            ),
        )
        contextual_bandit = contextual_bandit_utils.add_model(
            model=new_model,
            contextual_bandit=contextual_bandit,
        )


if __name__ == '__main__':
    main()
