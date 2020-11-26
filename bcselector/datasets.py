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


def load_mimic3(as_frame=True, discretize_data=True, **kwargs):
    """Load and return the mimic3 dataset.
    The mimic3 dataset is a medical dataset with multiple target variables.
    Dataset is avaliable at Physiobank [1]_.
    Costs of features were collected in article [2]_.

    =================   ==============
    Samples total                 6591
    Dimensionality                 306
    Target variables                10
    =================   ==============

    Parameters
    ----------
    as_frame : bool, default=True
        If True, the data is a pandas DataFrame including columns with
        appropriate names. The target is a pandas DataFrame with multiple target variables.
    discretize_data: bool, default=True
        If True, the returned data is discretized with sklearn.preprocessing.KBinsDiscretizer.
    **kwargs
        Arguments passed to sklearn.preprocessing.KBinsDiscretizer constructor.

    Returns
    -------
    data : {np.ndarray, pd.DataFrame} of shape (6591, 306)
        The data matrix. If `as_frame=True`, `data` will be a pd.DataFrame.
    target: {np.ndarray, pd.DataFrame} of shape (6591, 10)
        The binary classification target variable. If `as_frame=True`, `target` will be a pd.DataFrame.
    costs: {dict, list)
        Cost of every feature in data. If `as_frame=True`, `target` will be a dict.

    References
    ----------
    .. [1] MIMIC-III, a freely accessible critical care database. Johnson AEW, Pollard TJ, Shen L, Lehman LH, Feng M, Ghassemi M, Moody B, Szolovits P, Celi LA, and Mark RG. Scientific Data (2016). DOI: 10.1038/sdata.2016.35. Available at: http://www.nature.com/articles/sdata201635.
    .. [2] Paweł Teisseyre, Damien Zufferey, and Marta Słomka. Cost-sensitive classifier chains: Se-lecting low-cost features in multi-label classification.Pattern Recognition, 86, 09 2018.

    Examples
    --------
    >>> from bcselector.dataset import load_mimic3
    >>> data, target, costs = load_mimic3()
    """

    module_path = dirname(__file__)
    # Load data
    data = pd.read_csv(join(module_path, 'data', 'mimic3', 'mimic3.csv'))
    targets = pd.read_csv(join(module_path, 'data', 'mimic3', 'mimic3_targets.csv'))

    with open(join(module_path, 'data', 'mimic3', 'mimic3_costs.json'), 'r') as j:
        costs = json.load(j)

    if discretize_data:
        data_discretized = np.apply_along_axis(func1d=_discretize, axis=0, arr=data.values, **kwargs)
        data = pd.DataFrame(data_discretized, columns=data.columns)

    if as_frame:
        return data, targets, costs
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
    target: {np.ndarray, pd.DataFrame} of shape (6591, 10)
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
        data_discretized = np.apply_along_axis(func1d=_discretize, axis=0, arr=data.values, **kwargs)
        data = pd.DataFrame(data_discretized, columns=data.columns)

    if as_frame:
        return data, targets, costs
    else:
        return data.values, targets.values, list(costs.values())
