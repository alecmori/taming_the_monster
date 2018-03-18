The purpose of this repo is to provide a template of anyone wishing to evaluate contextual bandit systems. It provides three functions: **Data Generation**, **Training**, and **Evaluation**.

Generate Data
=============
The part of the repo that can be found at [taming_the_monster/generate_data](https://github.com/alecmori/taming_the_monster/tree/master/taming_the_monster/generate_data) as well as the [default-config.yaml](https://github.com/alecmori/taming_the_monster/blob/master/default-config.yaml#L4) file. If you do not have your own data to play with, you can generate your own data using the command `make data` (or run the commands found in the [Makefile](https://github.com/alecmori/taming_the_monster/blob/master/Makefile)). Descriptions of what each module does can be found in the docstrings.

Relevant Links
==============
[Unbiased Offline Evaluation of Contextual-bandit-based
News Article Recommendation Algorithms](https://arxiv.org/pdf/1003.5956.pdf)
[Taming the Monster: A Fast and Simple Algorithm for Contextual Bandits](https://arxiv.org/pdf/1402.0555.pdf)
