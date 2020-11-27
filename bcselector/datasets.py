import json
from os.path import dirname, join

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer


def _discretize(vector, **kwargs):
    """Discretizes vector with sklearn.preprocessing.KBinsDiscretizer.

    Parameters
    ----------
    vector : np.array
    kwargs
        Arguments passed to sklearn.preprocessing.KBinsDiscretizer constructor.

    Returns
    -------
    discretized_vector: np.array
        Discretized by **kwargs arguments method vector.
    """
    discretizer = KBinsDiscretizer(encode='ordinal', **kwargs)
    discretized_vector = discretizer.fit_transform(vector.reshape(-1, 1)).reshape(-1)
    return discretized_vector


def load_sample(as_frame=True):
    """Load and return the sample artificial dataset.

    =================   ==============
    Samples total                10000
    Dimensionality                  35
    Target variables                 1
    =================   ==============

    Parameters
    ----------
    as_frame : bool, default=True
        If True, the data is a pandas DataFrame including columns with
        appropriate names. The target is a pandas DataFrame with multiple target variables.

    Returns
    -------
    data : {np.ndarray, pd.DataFrame} of shape (10000, 35)
        The data matrix. If `as_frame=True`, `data` will be a pd.DataFrame.
    target: {np.ndarray, pd.Series} of shape (10000, 35)
        The binary classification target variable. If `as_frame=True`, `target` will be a pd.DataFrame.
    costs: {dict, list)
        Cost of every feature in data. If `as_frame=True`, `target` will be a dict.

    Examples
    --------
    >>> from bcselector.dataset import load_sample
    >>> data, target, costs = load_sample()
    """

    module_path = dirname(__file__)
    # Load data
    data = pd.read_csv(join(module_path, 'data', 'sample_data', 'sample_data.csv'))
    targets = pd.read_csv(join(module_path, 'data', 'sample_data', 'sample_target.csv'))

    with open(join(module_path, 'data', 'sample_data', 'sample_costs.json'), 'r') as j:
        costs = json.load(j)

    if as_frame:
        return data, targets['Class'], costs
    else:
        return data.values, targets.values, list(costs.values())


def load_hepatitis(as_frame=True, discretize_data=True, **kwargs):
    """Load and return the hepatitis dataset provided.
    The mimic3 dataset is a small medical dataset with single target variable.
    Dataset is collected from UCI repository [3]_.

    =================   ==============
    Samples total                  155
    Dimensionality                  19
    Target variables                 1
    =================   ==============

    Parameters
    ----------
    as_frame : bool, default=True
        If True, the data is a pandas DataFrame including columns with
        appropriate names. The target is a pandas DataFrame with multiple target variables.
    discretize_data: bool, default=True
        If True, the returned data is discretized with sklearn.preprocessing.KBinsDiscretizer.
    kwargs
        Arguments passed to sklearn.preprocessing.KBinsDiscretizer constructor.

    Returns
    -------
    data : {np.ndarray, pd.DataFrame} of shape (6591, 306)
        The data matrix. If `as_frame=True`, `data` will be a pd.DataFrame.
    target: {np.ndarray, pd.Series} of shape (6591, 10)
        The binary classification target variable. If `as_frame=True`, `target` will be a pd.DataFrame.
    costs: {dict, list)
        Cost of every feature in data. If `as_frame=True`, `target` will be a dict.

    References
    ----------
    .. [3] Dua, D. and Graff, C. (2019). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

    Examples
    --------
    >>> from bcselector.dataset import load_hepatitis
    >>> data, target, costs = load_hepatitis()
    """
    module_path = dirname(__file__)

    # Load data
    data = pd.read_csv(join(module_path, 'data', 'hepatitis', 'hepatitis.csv'))
    targets = pd.read_csv(join(module_path, 'data', 'hepatitis', 'hepatitis_target.csv'))

    with open(join(module_path, 'data', 'hepatitis', 'hepatitis_costs.json'), 'r') as j:
        costs = json.load(j)

    if discretize_data:
        data_colnames = data.columns
        n_bins = kwargs.get('n_bins', 10)
        col_to_discretize = data.nunique()[data.nunique() > n_bins].index
        col_not_changing = data.nunique()[data.nunique() <= n_bins].index

        data_discretized = np.apply_along_axis(func1d=_discretize, axis=0, arr=data[col_to_discretize].values, **kwargs)
        data = pd.concat([pd.DataFrame(data_discretized, columns=col_to_discretize), data[col_not_changing]], axis=1)
        data = data[data_colnames]

    if as_frame:
        return data, targets['Class'], costs
    else:
        return data.values, targets.values, list(costs.values())
