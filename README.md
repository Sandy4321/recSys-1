.. -*- mode: rst -*-

Machine Learning for Recommender Systems
========================================

The goal of this project is to apply machine learning techniques to recommender system problems.
Currently a single problem is under investigation:

- Hybrid recommender algorithm

Seethe `AUTHORS.rst <AUTHORS.rst>`_ file for a complete list of contributors.

In the following we provide a brief description of the problem.

Hybrid Recommender System
-------------------------
This project aims to exploit machine learning techniques for the development of a personalized hybrid recommender algorithm.

The idea is to learn a data-driven model that, given the user rating and the item attribute matrices, is able to predict the user rates.
The problem is formalized as a **supervised** regression problem where the targets are provided by the user rates.

This project is developed in python leveraging on freely available libraries.
