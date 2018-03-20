# -*- coding: utf-8 -*-
import numpy

from taming_the_monster import model


def filter_data(training_data, contextual_bandit):
    """TODO
    """
    num_examples, _, _ = training_data['possible_actions'].shape
    chosen_indices = [
        i
        for i in range(num_examples)
        if any(
            _chose_best_action(
                possible_actions=training_data['possible_actions'][i],
                chosen_action=training_data['chosen_actions'][i],
                policy=policy,
            )
            for policy in contextual_bandit
        )
    ]
    return {
        'possible_actions': numpy.take(
            training_data['possible_actions'],
            chosen_indices,
        ),
    }


def _chose_best_action(possible_actions, chosen_action, policy):
    """TODO
    """
    return possible_actions[
        numpy.argmax(
            a=model.score_actions(X=possible_actions, model=policy['model']),
        )
    ] == chosen_action
