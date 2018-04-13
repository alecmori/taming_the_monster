# -*- coding: utf-8 -*-
import numpy

from taming_the_monster.train import contextual_bandit_utils
from taming_the_monster.train import model_utils


def evaluate_contextual_bandit(contextual_bandit, data):
    """Algo 2 from the paper"""
    print(len(contextual_bandit))
    unbiased_reward = float(
        sum(data['rewards']),
    ) / len(data['rewards'])
    average_observed_reward = 0
    num_observed_values = 0
    for possible_actions, observed_action, reward in zip(
        data['possible_actions'],
        data['chosen_actions'],
        data['rewards'],
    ):
        p = _get_probability_of_choosing(
            contextual_bandit=contextual_bandit,
            possible_actions=possible_actions,
            observed_action=observed_action,
        )
        num_observed_values += p
        average_observed_reward += p * reward
    return (
        average_observed_reward / num_observed_values -
        unbiased_reward
    ) / unbiased_reward


def _get_probability_of_choosing(
    contextual_bandit, possible_actions, observed_action,
):
    """TODO"""
    return sum(
        policy['probability']
        for policy in contextual_bandit
        if contextual_bandit_utils.get_chosen_action_index(
            actions=possible_actions,
            chosen_action=observed_action,
        ) == numpy.argmax(
            model_utils.score_actions(
                X=possible_actions,
                model=policy['model'],
            ),
        )
    )
