# -*- coding: utf-8 -*-
"""
This is meant to simulate what would happen a real world epoch.

TODO: Flesh out description
"""
import numpy


def run_single_epoch(
        model, actual_policy, num_actions, num_features, num_examples,
):
    """Generates one epoch of fake data

    One day goes as followed
    1. Generate fake data coming into the system
    2. Use single model (NOT CONTEXTUAL BANDIT) to generate biased data
    3. Use an unknown behind-the-scenes algorithm to create observed
        reward based off action chosen

    :param model: The model trained offline on pieces of data.
    :type model_ensemble: keras.model
    :param actual_policy: A function that takes in feature vectors and
        spits out a rewards - the Contextual Bandit should be trying to
        learn this.
    :type actual_policy: func
    :param num_actions: The number actions to decide between for each
        example.
    :type num_actions: int
    :param num_features: The number of features each piece of data has.
    :type num_features: int
    :param num_examples: The number of examples to be generated in each
        epoch.
    :type num_examples: int

    :returns: The feature vectors for the possible actions you could
        have shown, the action you actually did show, and the observed
        reward for that action.
    :rtype: dict

        {
            'all_actions': list of the feature vectors of each
                possible action for each example.
            'chosen_actions': list of the chosen actions for
                each example.
            'chosen_action_rewards': list of floats which
                represent the real-world reward we say for each action.
        }
    """
    fake_data = _generate_fake_data(
        num_actions=num_actions,
        num_features=num_features,
        num_examples=num_examples,
    )
    scored_actions = model.predict(x=fake_data).reshape(
        (num_examples, num_actions),
    )
    structured_data = fake_data.reshape(
        (num_examples, num_actions, num_features),
    )
    structured_scores = actual_policy(X=fake_data).reshape(
        (num_examples, num_actions),
    )
    return {
        'all_actions': structured_data.tolist(),
        'chosen_actions': numpy.array(
            [
                structured_data[i, chosen_index, :]
                for i, chosen_index in enumerate(
                    numpy.argmax(scored_actions, axis=1),
                )
            ],
        ).tolist(),
        'chosen_action_rewards': numpy.max(structured_scores, axis=1).tolist(),
    }


def _generate_fake_data(num_examples, num_features, num_actions):
    """Generates fake data for the day

    :param num_examples: The number of fake data points
        you want to generate per day.
    :type num_examples: int
    :param num_actions: The number of fake actions to choose
        between per day.
    :type num_actions: int
    :param num_features: Number of features per actions
    :type num_features: dict

    :returns: The data generated for a given epoch - each row is a set
        of features
    :rtype: numpy.ndarray
    """
    return numpy.random.random(
        (
            num_examples * num_actions,
            num_features,
        ),
    )
