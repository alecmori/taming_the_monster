# -*- coding: utf-8 -*-
import numpy

from taming_the_monster.train import model_utils


def add_model(
    model, contextual_bandit, possible_actions, chosen_actions, Y, weights,
    min_probs,
):
    """TODO"""
    expected_reward = _get_expected_reward(
        model=model,
        possible_actions=possible_actions,
        chosen_actions=chosen_actions,
        Y=Y,
        weights=weights,
    )
    scaled_regret = _get_scaled_regret(
        expected_regret=max(
            policy['expected_reward'] - expected_reward
            for policy in contextual_bandit
        ),
        min_probs=min_probs,
    )
    variance_coefficients = _get_variance_coefficients(
        contextual_bandit=contextual_bandit,
        model=model,
        possible_actions=possible_actions,
        scaled_regret=scaled_regret,
        min_probs=min_probs,
    )
    if variance_coefficients['D'] > 0:
        return _rescale_probability(
            contextual_bandit=contextual_bandit + [
                {
                    'expected_reward': expected_reward,
                    'probability': 0.0,
                    'model': '',
                },
            ],
        )
    else:
        return contextual_bandit


def _get_expected_reward(model, possible_actions, chosen_actions, Y, weights):
    """TODO"""
    for actions in possible_actions:
        model_utils.score_actions(X=actions, model=model)


def _get_scaled_regret(expected_regret, min_probs):
    """TODO
    Explained in OP under Algorithm 1
    """
    return expected_regret / (100 * numpy.average(min_probs))


def _get_variance_coefficients(
    contextual_bandit, model, possible_actions, scaled_regret, min_probs,
):
    """TODO"""
    pass


def _rescale_probability(contextual_bandit):
    """TODO"""
    total_weight = sum(policy['probability'] for policy in contextual_bandit)
    return [
        {
            'expected_reward': policy['expected_reward'],
            'probability': policy['probability'] / total_weight,
            'model': policy['model'],
        }
        for policy in contextual_bandit
    ]
