# -*- coding: utf-8 -*-
import json

import numpy


def iterate_data():
    """Yields numpy iterator
    """
    # Yield for each epoch
    for row in open('data/train_data'):
        d = json.loads(row)
        yield {
            # Shape = (num_examples, num_actions_possible, num_features)
            'possible_actions': numpy.array(d['all_actions']),
            # Shape = (num_examples, num_features)
            'chosen_actions': numpy.array(d['chosen_actions']),
            # Shape = (num_examples, 1)
            'rewards': numpy.array(d['chosen_action_rewards']),
        }
