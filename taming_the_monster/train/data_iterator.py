# -*- coding: utf-8 -*-


def iterate_data():
    """Yields numpy iterator
    """
    # Yield for each epoch
    yield {
        # Shape = (num_examples, num_actions_possible, num_features)
        'possible_actions': [],
        # Shape = (num_examples, num_features)
        'chosen_actions': [],
        # Shape = (num_examples, 1)
        'rewards': [],
    }
