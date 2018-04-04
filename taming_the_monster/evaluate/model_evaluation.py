# -*- coding: utf-8 -*-
import numpy

from taming_the_monster.train import contextual_bandit_utils
from taming_the_monster.train import model_utils

NUM_TRIALS = 50


def evaluate_contextual_bandit(contextual_bandit, data):
    """Algo 2 from the paper"""
    print(len(contextual_bandit))
    return numpy.average(
        [
            _simulate_single_contextual_bandit(
                contextual_bandit=contextual_bandit,
                data=data,
            )
            for _ in range(NUM_TRIALS)
        ],
    )


def _simulate_single_contextual_bandit(contextual_bandit, data):
    observed_reward = 0
    num_observed_values = 0
    for possible_actions, observed_action, reward in zip(
        data['possible_actions'],
        data['chosen_actions'],
        data['rewards'],
    ):
        policy = numpy.random.choice(
            a=contextual_bandit,
            p=[policy['probability'] for policy in contextual_bandit],
        )
        if contextual_bandit_utils.get_chosen_action_index(
            actions=possible_actions,
            chosen_action=observed_action,
        ) == numpy.argmax(
            model_utils.score_actions(
                X=possible_actions,
                model=policy['model'],
            ),
        ):
            observed_reward += reward
            num_observed_values += 1
    return observed_reward / num_observed_values
