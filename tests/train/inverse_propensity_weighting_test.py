# -*- coding: utf-8 -*-
import mock
import pytest

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


@pytest.mark.usefixtures('patch_get_min_prob', 'patch_get_prob_of_choosing')
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

    @pytest.mark.usefixtures('patch_get_prob_of_choosing')
    def test_calls_get_min_prob(
            self, patch_get_min_prob, fake_minimum_probabilities,
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
            self, patch_get_min_prob, patch_get_prob_of_choosing, rewards,
            min_probs, expected,
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
        # And the weighted_rewards should be <expected>
        assert return_value['weighted_rewards'] == expected

    def test_chooses_largest_probability(
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
        # And the rewards should be weighted by the largest probability
        assert return_value['weighted_rewards'] == [2., 2.5]
