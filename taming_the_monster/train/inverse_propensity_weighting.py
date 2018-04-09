# -*- coding: utf-8 -*-
import numpy

from taming_the_monster.train import contextual_bandit_utils


FAILURE_PROB = 0.01


def get_propensity_info(
        possible_actions, chosen_actions, contextual_bandit, epoch,
        score_actions, rewards,
):
    """Gets the reweighted rewards that a policy can learn

    For further reading, read Algorithm 1 from `Taming the Monster`.

    :param possible_actions: For each training example, a list of all
        possible feature vectors that were considered for before
        choosing to show one.
    :type possible_actions: list (of list of feature vectors)
    :param chosen_actions: The observed actions; must be one of the
        `possible_actions`.
    :type chosen_actions:
    :param contextual_bandit: The contextual bandit that has been
        trained thus far.
    :type contextual_bandit: list (of dicts)
    :param epoch: A monotone increasing value representing which epoch
        we are in.
    :type epoch: int
    :param score_actions: A user-provided function that will score a
        bunch of user-provided features given a model and a feature
        matrix.
    :type score_actions: func
    :param rewards: Given the list of chosen actions, one the observed
        rewrad for each.
    :type rewards: list (of floats)

    :returns: Both the minimum probabilities per action and the
        new weighted rewards.
    :rtype: dict
    """
    minimum_probs = _get_min_prob(
        possible_actions=possible_actions,
        contextual_bandit=contextual_bandit,
        epoch=epoch,
        failure_prob=FAILURE_PROB,
    )
    return {
        'weighted_rewards': _normalize(
            rewards=[
                reward / max(
                    min_prob,
                    _get_prob_of_choosing(
                        action_list=action_list,
                        chosen_action=chosen_action,
                        contextual_bandit=contextual_bandit,
                        score_actions=score_actions,
                    ),
                )
                for action_list, chosen_action, min_prob, reward
                in zip(
                    possible_actions,
                    chosen_actions,
                    minimum_probs,
                    rewards,
                )
            ],
        ),
        'min_probs': minimum_probs,
    }


def _get_min_prob(
        possible_actions, contextual_bandit, epoch, failure_prob,
):
    """Definition of minimum probabilty from `Taming the Monster`

    For further reading, read Algorithm 1 (specifically the part about
    minimum probability) from `Taming the Monster`.

    :param possible_actions: For each training example, a list of all
        possible feature vectors that were considered for before
        choosing to show one.
    :type possible_actions: list (of list of feature vectors)
    :param contextual_bandit: The contextual bandit that has been
        trained thus far.
    :type contextual_bandit: list (of dicts)
    :param epoch: A monotone increasing value representing which epoch
        we are in.
    :type epoch: int
    :param failure_prob: The probability of "failure"; similar to
        willingness to explore.
    :type failure_prob: float

    :returns: A list of the minimum possible probabilities per training
        example; used in weighting the "new" reward.
    :rtype: list (of floats)

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


def _get_prob_of_choosing(
        action_list, chosen_action, contextual_bandit, score_actions,
):
    """Gets the actual probability of being chosen by the bandit

    For further reading, read Algorithm 1 from `Taming the Monster`.

    :param action_list: A list of all possible feature vectors that
        were considered for before choosing to show one.
    :type possible_actions: list (of feature vectors)
    :param chosen_action: The observed action; must be one of the
        `possible_actions`.
    :type chosen_actions: feature vector (given by user)
    :param contextual_bandit: The contextual bandit that has been
        trained thus far.
    :type contextual_bandit: list (of dicts)
    :param score_actions: A user-provided function that will score a
        bunch of user-provided features given a model and a feature
        matrix.
    :type score_actions: func

    :returns: The probability the `chosen_action` has of being picked
        by the contextual bandit.
    :rtype: float
    """
    chosen_action_index = contextual_bandit_utils.get_chosen_action_index(
        actions=action_list,
        chosen_action=chosen_action,
    )
    return sum(
        policy['probability']
        for policy in contextual_bandit
        if numpy.argmax(
            score_actions(X=action_list, model=policy['model']),
        ) == chosen_action_index
    )


def _normalize(rewards):
    """Normalize the weighted rewards between 0 and 1 for training

    :param rewards: The rewards after being weighted by their
        propensities.
    :type rewards: list (of floats)

    :returns: The rewards still proportionally the same, but bounded
        between 0 and 1 (inclusive).
    :rtype: list (of floats)
    """
    return [reward / max(rewards) for reward in rewards]
