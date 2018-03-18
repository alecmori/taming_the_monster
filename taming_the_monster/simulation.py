# -*- coding: utf-8 -*-
import json

import numpy
import staticconf

from taming_the_monster import model
from taming_the_monster import policy
from taming_the_monster import run_single_epoch

CONFIG = 'default-config.yaml'


def main():
    """TODO DESCRIBE THIS
    """
    staticconf.YamlConfiguration(CONFIG)
    if staticconf.read('generate_data.enabled'):
        generate_biased_data(
            policy=policy.get_policy(
                complexity=staticconf.read('policy.complexity'),
                num_features=staticconf.read('model.num_features'),
                mean=staticconf.read('policy.mean'),
                std=staticconf.read('policy.std'),
            ),
        )


def generate_biased_data(policy):
    """TODO DESCRIBE THIS
    """
    single_model = _create_initial_model(policy=policy)
    epoch_history = []
    for _ in range(staticconf.read('generate_data.num_epochs')):
        epoch_info = run_single_epoch.run_single_epoch(
            model=single_model,
            actual_policy=policy,
            num_actions=staticconf.read('generate_data.num_actions'),
            num_features=staticconf.read('model.num_features'),
            num_examples=staticconf.read('generate_data.num_examples'),
        )
        epoch_history.append(epoch_info)
    if staticconf.read('generate_data.model_output'):
        # TODO: Implement this shit
        pass
    with open(staticconf.read('generate_data.output_file'), 'w+') as fh:
        for epoch_info in epoch_history:
            fh.write(json.dumps(epoch_info))
            fh.write('\n')


def _create_initial_model(policy):
    """TODO DESCRIBE THIS
    """
    fake_data = numpy.random.random(
        (
            staticconf.read('generate_data.num_examples') * 5,
            staticconf.read('model.num_features'),
        ),
    )
    return model.train_single_model(
        X=fake_data,
        Y=policy(X=fake_data),
    )


if __name__ == '__main__':
    main()
