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
    ) as mock_instance:
        yield mock_instance


@pytest.mark.usefixtures('patch_get_min_prob', 'patch_get_prob_of_choosing')
class TestGetPropensityInfo(object):

    def test_pytest(self):
        assert 1 == 1
