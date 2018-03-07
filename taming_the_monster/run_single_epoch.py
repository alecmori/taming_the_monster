# -*- coding: utf-8 -*-
import numpy

DATA_GENERATING_FUNCTIONS = {
    'binary': lambda x: numpy.random.randint(0, 2, x),
    'normalized': numpy.random.standard_normal,
}
NUM_DATA_POINTS = 100000
NUM_FEATURES = 1000


def run_single_epoch(contextual_bandit, **kwargs):
    """Generates one epoch of fake data

    One day goes as followed
    1. Generate fake data coming into the system
    2. Use the contextual bandit to choose an action based off that
        data
    3. Use an unknown behind-the-scenes algorithm to create observed
        reward based off action chosen
    4. Also log the best possible reward from all the actions (Used
        to measure regret)

    """
    fake_data = _generate_fake_data(**kwargs)
    return fake_data


def _generate_fake_data(**kwargs):
    """Generates fake data for the day

    :param num_data_points: (Optional) The number of fake data points
        you want to generate per day.
    :type num_data_points: int
    :param data_description: (Optional) Descriptions of the data, see
        README.md for config descriptions.
    :type data_description: dict

    :returns: The data generated for a given epoch - each row is a set
        of features
    :rtype: numpy.ndarray
    """
    if 'num_data_points' in kwargs:
        num_data_points = kwargs['num_data_points']
    else:
        num_data_points = NUM_DATA_POINTS
    return numpy.vstack(
        numpy.concatenate(
            [
                function(
                    kwargs.get(
                        'data_description',
                        {},
                    ).get(feature_type, NUM_FEATURES),
                )
                for feature_type, function in DATA_GENERATING_FUNCTIONS.items()
            ],
        )
        for _ in range(num_data_points)
    )


if __name__ == '__main__':
    pass
