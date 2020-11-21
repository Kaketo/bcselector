==========
Bcselector
==========
.. image:: https://raw.githubusercontent.com/Kaketo/bcselector/master/docs/img/logo_small.png

.. image:: https://img.shields.io/badge/python-3.7-blue.svg
    :target: http://badge.fury.io/py/bcselector
.. image:: https://badge.fury.io/py/bcselector.svg
    :target: https://badge.fury.io/py/bcselector
.. image:: https://travis-ci.com/Kaketo/bcselector.svg?branch=master
    :target: https://travis-ci.com/Kaketo/bcselector
.. image:: https://codecov.io/gh/Kaketo/bcselector/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/Kaketo/bcselector
.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
  :target: https://opensource.org/licenses/MIT

* Documentation: https://kaketo.github.io/bcselector.
* Repository: https://github.com/kaketo/bcselector.

What is it?
-----------
Feature selection is a crucial problem in many machine learning tasks. Usually the considered
variables are cheap to collect and store but in some situations the acquisition of feature values
can be problematic. For example, when predicting the occurrence of the disease we may consider
the results of some diagnostic tests which can be very expensive.
The existing feature selection methods usually ignore costs associated with the considered
features. The goal of cost- sensitive feature selection is to select a subset of features which allow
to predict the target variable (e.g. occurrence of the diseases) successfully within the assumed
budget.

The main purpose of this package is to provide filter methods of feature selection based
on information theory and to propose new variants of these methods considering feature costs.


Installation
------------

bcselector can be installed from [PyPI](https://pypi.org/project/bcselector): ::

    pip install bcselector