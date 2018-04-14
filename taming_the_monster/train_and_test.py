# -*- coding: utf-8 -*-
from taming_the_monster.evaluate import get_test_data
from taming_the_monster.evaluate import model_evaluation
from taming_the_monster.train import data_iterator
from taming_the_monster.train import model_utils
from taming_the_monster.train import train_contextual_bandit


def main():
    contextual_bandit = train_contextual_bandit.train_contextual_bandit(
        iterate_data=data_iterator.iterate_data,
        train_model=model_utils.train_model,
        score_actions=model_utils.score_actions,
    )
    evaluated_metrics = model_evaluation.evaluate_contextual_bandit(
        contextual_bandit=contextual_bandit,
        data=get_test_data.get_test_data(),
    )
    print(evaluated_metrics)


if __name__ == '__main__':
    main()
