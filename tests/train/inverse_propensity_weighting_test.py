# -*- coding: utf-8 -*-
import mock
import pytest

from taming_the_monster.train import contextual_bandit_utils
from taming_the_monster.train import inverse_propensity_weighting


@pytest.fixture
def patch_get_propensity_info():
    with mock.patch.object(
        inverse_propensity_weighting,
        'get_propensity_info',
        autospec=True,
    ) as mock_instance:
        yield mock_instance


@pytest.fixture
def patch_get_min_prob():
    with mock.patch.object(
        inverse_propensity_weighting,
        '_get_min_prob',
        autospec=True,
    ) as mock_instance:
        yield mock_instance


@pytest.fixture
def patch_get_prob_of_choosing():
    with mock.patch.object(
        inverse_propensity_weighting,
        '_get_prob_of_choosing',
        autospec=True,
        return_value=0.,
    ) as mock_instance:
        yield mock_instance


@pytest.fixture
def patch_normalize():
    with mock.patch.object(
        inverse_propensity_weighting,
        '_normalize',
        autospec=True,
    ) as mock_instance:
        yield mock_instance


@pytest.mark.usefixtures('patch_get_prob_of_choosing')
class TestGetPropensityInfo(object):

    @pytest.fixture
    def fake_possible_actions(self):
        return [mock.sentinel.action_1, mock.sentinel.action_2]

    @pytest.fixture
    def fake_chosen_actions(self):
        return [mock.sentinel.action_3, mock.sentinel.action_4]

    @pytest.fixture
    def fake_minimum_probabilities(self):
        return [0.5, 0.2]

    @pytest.fixture
    def fake_rewards(self):
        return [0, 1]

    def _call(self, **kwargs):
        params = {
            'possible_actions': self.fake_possible_actions(),
            'chosen_actions': self.fake_chosen_actions(),
            'contextual_bandit': mock.sentinel.contextual_bandit,
            'epoch': mock.sentinel.epoch,
            'score_actions': mock.sentinel.score_actions,
            'rewards': self.fake_rewards(),
        }
        params.update(kwargs)
        return inverse_propensity_weighting.get_propensity_info(**params)

    def test_calls_get_min_prob(
            self, patch_get_min_prob, patch_normalize,
            fake_minimum_probabilities,
    ):
        # Given we get no errors in getting the minimum probabilities
        patch_get_min_prob.return_value = fake_minimum_probabilities
        # When we try to get the propensity_info
        self._call()
        # Then we will have gotten the minimum probabilities once
        assert patch_get_min_prob.call_count == 1

    @pytest.mark.parametrize(
        argnames=['rewards', 'min_probs', 'expected'],
        argvalues=[
            ([1.], [0.2], [5.]),
            ([0.], [0.000001], [0.]),
            ([1., 0.], [0.04, 0.05], [25., 0.]),
        ],
        ids=['one_reward', 'zero_reward', 'mixed_rewards'],
    )
    def test_reweights_reward_correctly(
            self, patch_get_min_prob, patch_get_prob_of_choosing,
            patch_normalize, rewards, min_probs, expected,
    ):
        # Given the probability of choosing is 0
        patch_get_prob_of_choosing.return_value = 0.
        # And min_probabilities retrieved are <min_probs>
        patch_get_min_prob.return_value = min_probs
        # When we get propensity information with <rewards>
        return_value = self._call(
            possible_actions=[mock.sentinel for _ in range(len(rewards))],
            chosen_actions=[mock.sentinel for _ in range(len(rewards))],
            rewards=rewards,
        )
        # Then the min_probs should match <min_probs>
        assert return_value['min_probs'] == min_probs
        # And the unnormalized weighted_rewards should be <expected>
        args, kwargs = patch_normalize.call_args
        assert kwargs['rewards'] == expected

    def test_chooses_largest_probability(
            self, patch_get_min_prob, patch_get_prob_of_choosing,
            patch_normalize,
    ):
        # Given that we have differing probability weights
        min_probs = [0.5, 0.25]
        patch_get_min_prob.return_value = min_probs
        patch_get_prob_of_choosing.side_effect = [0.2, 0.4]
        # When we get the weighted rewards
        return_value = self._call(
            possible_actions=[mock.sentinel for _ in range(len(min_probs))],
            chosen_actions=[mock.sentinel for _ in range(len(min_probs))],
            rewards=[1., 1.],
        )
        # Then the min_probs should match <min_probs>
        assert return_value['min_probs'] == min_probs
        # And the unnormalized rewards should be weighted correctly
        args, kwargs = patch_normalize.call_args
        assert kwargs['rewards'] == [2., 2.5]

    def test_normalizes(
            self, patch_get_min_prob, patch_get_prob_of_choosing,
    ):
        # Given that we have differing probability weights
        min_probs = [0.5, 0.25]
        patch_get_min_prob.return_value = min_probs
        patch_get_prob_of_choosing.side_effect = [0.2, 0.4]
        # When we get the weighted rewards
        return_value = self._call(
            possible_actions=[mock.sentinel for _ in range(len(min_probs))],
            chosen_actions=[mock.sentinel for _ in range(len(min_probs))],
            rewards=[1., 1.],
        )
        # Then the min_probs should match <min_probs>
        assert return_value['min_probs'] == min_probs
        # And the results should be normalized
        assert return_value['weighted_rewards'] == [0.8, 1.0]


class TestGetMinProb(object):

    def _call(self, **kwargs):
        params = {
            'possible_actions': mock.sentinel.possible_actions,
            'contextual_bandit': mock.sentinel.contextual_bandit,
            'epoch': mock.sentinel.epoch,
            'failure_prob': mock.sentinel.failure_prob,
        }
        params.update(kwargs)
        return inverse_propensity_weighting._get_min_prob(**params)

    @pytest.mark.parametrize(
        argnames=['num_possible_actions', 'expected'],
        argvalues=[(4, 0.125), (5, 0.1)],
        ids=['four_actions', 'five_actions'],
    )
    def test_formula_epoch_zero(self, num_possible_actions, expected):
        # Given we are at epoch zero
        # And we have <num_possible_actions> for 10 values
        possible_actions = [
            [
                mock.sentinel.action_list
                for action_list in range(num_possible_actions)
            ]
            for _ in range(10)
        ]
        # When we try to find the minimum probability
        result = self._call(
            possible_actions=possible_actions,
            epoch=0,
        )
        # Then all min_probs should be the same
        assert len(set(result)) == 1
        # And they should all be <expected>
        assert next(min_prob for min_prob in result) == expected

    @pytest.mark.parametrize(
        argnames=[
            'num_possible_actions',
            'num_policies',
            'epoch',
            'failure_prob',
            'expected',
        ],
        argvalues=[
            (
                10,
                4,
                1,
                0.02,
                [0.05],
            ),
            (
                8,
                3,
                1,
                0.01,
                [0.0625],
            ),
            (
                14,
                5,
                10,
                0.02,
                [0.03571428571428571],
            ),
        ],
        ids=['small_epoch_1', 'small_epoch_2', 'large_epoch_1'],
    )
    def test_formula_epoch_non_zero(
            self, num_possible_actions, num_policies, epoch, failure_prob,
            expected,
    ):
        # Given some parameters
        # When we try to get the epoch 1 formula
        result = self._call(
            possible_actions=[
                [
                    mock.sentinel.possible_actions
                    for _ in range(num_possible_actions)
                ],
            ],
            contextual_bandit=[
                [
                    mock.sentinel.policy
                    for _ in range(num_policies)
                ],
            ],
            epoch=epoch,
            failure_prob=failure_prob,
        )
        # Then we should get <expected>
        assert result == expected


def TestGetProbOfChoosing(object):

    @pytest.fixture
    def patch_get_chosen_action_index(self):
        with mock.patch.object(
            contextual_bandit_utils,
            'get_chosen_action_index',
            autospec=True,
        ) as mock_instance:
            yield mock_instance

    def _call(self, **kwargs):
        params = {
            'action_list': mock.sentinel.action_list,
            'chosen_action': mock.sentinel.chosen_action,
            'contextual_bandit': mock.sentinel.contextual_bandit,
            'score_actions': mock.sentinel.score_actions,
        }
        params.update(kwargs)
        return inverse_propensity_weighting._get_prob_of_choosing(**params)


def TestNormalize(object):

    def _call(self, **kwargs):
        params = {'rewards': mock.sentinel.rewards}
        params.update(kwargs)
        return inverse_propensity_weighting.normalize(**params)

    @pytest.mark.parametrize(
        argnames=['rewards', 'expected'],
        argvalues=[
            ([0., 1., 0.5], [0., 1., 0.5]),
            ([0., 4., 0.2], [0., 1., 0.05]),
        ],
        ids=['normalize_with_one', 'normalize_with_greater_than_one'],
    )
    def test_normalize(self, rewards, expected):
        # When we normalize <rewards>
        results = self._call(rewards=rewards)
        # Then we expect <expected>
        assert results == expected
