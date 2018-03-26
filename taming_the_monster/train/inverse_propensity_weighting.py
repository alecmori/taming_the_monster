# -*- coding: utf-8 -*-
import numpy
import staticconf

from taming_the_monster.train import model_utils


def get_propesnity_info(
        possible_actions, chosen_actions, contextual_bandit, epoch,
):
    """TODO"""
    minimum_probs = _get_min_prob(
        possible_actions=possible_actions,
        contextual_bandit=contextual_bandit,
        epoch=epoch,
        failure_prob=staticconf.read(
            'train_contextual_bandit.failure_probability',
        ),
    )
    return {
        'weights': numpy.array(
            1. / max(
                min_prob,
                _get_prob_of_choosing(
                    action_list=action_list,
                    chosen_action=chosen_action,
                    contextual_bandit=contextual_bandit,
                ),
            )
            for action_list, chosen_action, min_prob
            in zip(possible_actions, chosen_actions, minimum_probs)
        ),
        'min_probs': minimum_probs,
    }


def _get_min_prob(
        possible_actions, contextual_bandit, epoch, failure_prob,
):
    """TODO
    as defined Algorithm 1
    """
    return [
        min(
            1. / (2 * len(action_list)),
            numpy.sqrt(
                numpy.log(
                    16 * epoch**2 * len(contextual_bandit) / failure_prob,
                ) / (len(action_list) * epoch),
            ),
        )
        if epoch > 0
        else 1. / (2 * len(action_list))
        for action_list in possible_actions
    ]


def _get_prob_of_choosing(action_list, chosen_action, contextual_bandit):
    """TODO"""
    chosen_action_index = next(
        numpy.where(action_list == chosen_action),
    )
    return sum(
        policy['probability']
        for policy in contextual_bandit
        if numpy.argmax(
            model_utils.score_actions(X=action_list, model=policy['model']),
        ) == chosen_action_index
    )
