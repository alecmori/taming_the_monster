# -*- coding: utf-8 -*-
import numpy


def add_model(
    model, contextual_bandit, possible_actions, chosen_actions,
    weighted_rewards, min_probs, score_actions,
):
    """Determines whether or not the new model is added to the bandit.

    Implementation of Algorithm 2 (Section 3.2) from the 'Taming the
    Monster' paper. Essentially, it checks whether or not the newly
    trained model picks the same things as the other models in your
    contextual bandit.
    If it does agree with the models in your ensemble, don't add this
    new model in - it adds nothing of value.
    If it doesn't, add it into your ensemble - it is capturing
    something different than all your current models and should be
    emphasized.

    :param model:
    :type model:
    :param contextual_bandit:
    :type contextual_bandit:
    :param possible_actions:
    :type possible_actions:
    :param chosen_actions:
    :type chosen_actions:
    :param weighted_rewards:
    :type weighted_rewards:
    :param min_probs:
    :type min_probs:
    :param score_actions:
    :type score_actions: func

    :returns:
    :rtype:
    """
    model_choices = _get_model_choices(
        model=model,
        possible_actions=possible_actions,
        score_actions=score_actions,
    )
    expected_reward = _get_expected_reward(
        model_choices=model_choices,
        possible_actions=possible_actions,
        chosen_actions=chosen_actions,
        weighted_rewards=weighted_rewards,
    )
    scaled_regret = _get_scaled_regret(
        expected_regret=max(
            [
                policy['expected_reward'] - expected_reward
                for policy in contextual_bandit
            ],
        ) if contextual_bandit else 0,
        min_probs=min_probs,
    )
    variance_coefficients = _get_variance_coefficients(
        contextual_bandit=contextual_bandit,
        possible_actions=possible_actions,
        scaled_regret=scaled_regret,
        min_probs=min_probs,
        model_choices=model_choices,
        score_actions=score_actions,
    )
    print(variance_coefficients)
    if variance_coefficients['D'] >= 0:
        print('Adding Model')
        return _rescale_probability(
            contextual_bandit=contextual_bandit + [
                {
                    'expected_reward': expected_reward,
                    'probability': _get_new_model_probability(
                        V=variance_coefficients['V'],
                        S=variance_coefficients['S'],
                        D=variance_coefficients['D'],
                        possible_actions=possible_actions,
                        min_probs=min_probs,
                    ),
                    'model': model,
                },
            ],
        )
    else:
        return contextual_bandit


def _get_model_choices(model, possible_actions, score_actions):
    """TODO"""
    return [
        numpy.argmax(score_actions(X=actions, model=model))
        for actions in possible_actions
    ]


def _get_expected_reward(
        model_choices, possible_actions, chosen_actions, weighted_rewards,
):
    """TODO"""
    return numpy.sum(
        reward
        for model_choice, actions, chosen_action, reward in zip(
            model_choices,
            possible_actions,
            chosen_actions,
            weighted_rewards,
        )
        if get_chosen_action_index(
            actions=actions,
            chosen_action=chosen_action,
        ) == model_choice
    ) / len(possible_actions)


def get_chosen_action_index(actions, chosen_action):
    """TODO"""
    return numpy.where(actions == chosen_action)[0][0]


def _get_scaled_regret(expected_regret, min_probs):
    """TODO
    Explained in OP under Algorithm 1
    """
    return expected_regret / (100 * numpy.average(min_probs))


def _get_variance_coefficients(
    contextual_bandit, possible_actions, scaled_regret, min_probs,
    model_choices, score_actions,
):
    """TODO"""
    model_variances = _get_model_variances(
        contextual_bandit=contextual_bandit,
        model_choices=model_choices,
        possible_actions=possible_actions,
        min_probs=min_probs,
        score_actions=score_actions,
    )
    average_variance = numpy.average(model_variances)
    return {
        'V': average_variance,
        'S': numpy.average(numpy.power(model_variances, 2)),
        'D': average_variance - (
            scaled_regret +
            2 * _get_num_actions(possible_actions=possible_actions)
        ),
    }


def _get_num_actions(possible_actions):
    """TODO"""
    return numpy.average([len(actions) for actions in possible_actions])


def _get_model_variances(
    contextual_bandit, model_choices, possible_actions, min_probs,
    score_actions,
):
    """TODO"""
    return [
        1. / max(
            sum(
                policy['probability']
                for policy, model_choice in zip(
                    contextual_bandit,
                    model_choices,
                )
                if numpy.argmax(
                    score_actions(
                        X=actions,
                        model=policy['model'],
                    ),
                ) == model_choice
            ),
            min_prob,
        )
        for actions, min_prob in zip(possible_actions, min_probs)
    ]


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


def _get_new_model_probability(V, S, D, possible_actions, min_probs):
    """TODO"""
    numerator = V + D
    num_actions = _get_num_actions(possible_actions=possible_actions)
    denominator = 2. * (1. - num_actions *
                        _get_min_prob(min_probs=min_probs)) * S
    return numerator / denominator


def _get_min_prob(min_probs):
    """TODO"""
    return numpy.average(min_probs)
