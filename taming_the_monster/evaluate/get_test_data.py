# -*- coding: utf-8 -*-
import numpy
from sklearn import datasets

NUM_EXAMPLES = 1000
NUM_FEATURES = 2500
MAX_NUM_ACTIONS = 10
MIN_NUM_ACTIONS = 3


def get_unbiased_data(possible_actions_and_rewards):
    actions, rewards = possible_actions_and_rewards
    index = numpy.random.choice(
        a=range(len(actions)),
    )
    return {
        'possible_actions': actions,
        'chosen_action': actions[index],
        'reward': rewards[index],
    }


def get_test_data():
    """
    """
    examples = [
        get_unbiased_data(
            possible_actions_and_rewards=datasets.make_gaussian_quantiles(
                n_samples=numpy.random.choice(
                    a=range(MIN_NUM_ACTIONS, MAX_NUM_ACTIONS),
                ),
                n_features=NUM_FEATURES,
                n_classes=2,
            ),
        )
        for _ in range(NUM_EXAMPLES)
    ]
    return {
        # Shape = (num_examples, num_actions_possible, num_features)
        'possible_actions': [
            example['possible_actions'] for example in examples
        ],
        # Shape = (num_examples, num_features)
        'chosen_actions': [
            example['chosen_action'] for example in examples
        ],
        # Shape = (num_examples, 1)
        'rewards': [
            example['reward'] for example in examples
        ],
    }
