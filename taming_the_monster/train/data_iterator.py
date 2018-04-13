# -*- coding: utf-8 -*-
import numpy
from sklearn import datasets

NUM_EPOCHS = 25
NUM_EXAMPLES = 1000
NUM_FEATURES = 2500
MAX_NUM_ACTIONS = 10
MIN_NUM_ACTIONS = 3


def get_biased_data(possible_actions_and_rewards, epoch):
    actions, rewards = possible_actions_and_rewards
    index = numpy.random.choice(
        a=range(len(actions)),
        p=[
            float(reward + 1. / epoch) /
            (sum(rewards) + 1. / epoch * len(rewards))
            for reward in rewards
        ] if epoch else None,
    )
    return {
        'possible_actions': actions,
        'chosen_action': actions[index],
        'reward': rewards[index],
    }


def iterate_data():
    """Yields numpy iterator
    """
    # Yield for each epoch
    for epoch in range(NUM_EPOCHS):
        examples = [
            get_biased_data(
                possible_actions_and_rewards=datasets.make_gaussian_quantiles(
                    n_samples=numpy.random.choice(
                        a=range(MIN_NUM_ACTIONS, MAX_NUM_ACTIONS),
                    ),
                    n_features=NUM_FEATURES,
                    n_classes=2,
                ),
                epoch=epoch,
            )
            for _ in range(NUM_EXAMPLES)
        ]
        yield {
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
