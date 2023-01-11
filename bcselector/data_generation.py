import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer


class _BasicDataGenerator():
    def __init__(self):
        self.n_rows = None
        self.n_cols = None
        self.seed = None

    def generate(self, n_rows=100, n_cols=10, seed=None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        if seed is None:
            self.seed = np.random.randint(0, 420000)
        else:
            self.seed = seed
        np.random.seed(self.seed)


class MatrixGenerator(_BasicDataGenerator):
    def __init__(self):
        super().__init__()

    def _generate_basic_dataset(self, loc=0, scale=1):
        X = np.random.normal(loc=loc, scale=scale, size=(self.n_rows, self.n_cols))
        y = np.random.binomial(1, np.exp(X.sum(axis=1))/(1+np.exp(X.sum(axis=1))))
        return X, y

    def _generate_noise(self, sigma, loc=0):
        noise_sigmas = np.repeat(sigma, self.n_cols)
        noise = np.random.normal(loc=loc, scale=noise_sigmas, size=(self.n_rows, self.n_cols))
        return noise

    def generate(self, n_rows=100, n_basic_cols=10, loc=0,  noise_sigmas=None, basic_cost=1, seed=None, discretize_method='uniform', discretize_bins=10):
        assert isinstance(n_rows, int), "Argument `n_rows` must be int."
        assert isinstance(n_basic_cols, int), "Argument `n_cols` must be int."
        assert isinstance(loc, int), "Argument `loc` must be int or float."
        assert discretize_method in ['uniform', 'quantile', 'kmeans'], "Argument `discretize_method` must be in ['uniform', 'quantile', 'kmeans']"
        assert discretize_bins > 1 and isinstance(discretize_bins, int), "Argument `discretize_bins` must be int greater then 1."

        super().generate(n_rows, n_basic_cols, seed)

        self.loc = loc
        self.noise_sigmas = noise_sigmas

        # Generate basic features
        X_basic, y = self._generate_basic_dataset(loc=0, scale=1)
        costs = [basic_cost for i in range(self.n_cols)]
        if self.noise_sigmas is None:
            return X_basic, y, costs
        # Generate perturbed features
        X = X_basic.copy()
        for noise_sigma in self.noise_sigmas:
            noise = self._generate_noise(sigma=noise_sigma/(len(noise_sigmas)), loc=0)
            X_transformed = X_basic + noise
            X = np.concatenate((X, X_transformed), axis=1)
            costs = costs + [1/(noise_sigma + basic_cost) for i in range(self.n_cols)]

        # Discretize data
        def discretize(vector, **kwargs):
            discretizer = KBinsDiscretizer(encode='ordinal', **kwargs)
            vector = discretizer.fit_transform(vector.reshape(-1, 1)).reshape(-1)
            return vector

        X = np.apply_along_axis(func1d=discretize, axis=0, arr=X, n_bins=discretize_bins, strategy=discretize_method)

        return X, y, costs


class DataFrameGenerator(MatrixGenerator):
    def __init__(self):
        super().__init__()

    def _generate_colnames(self, n):
        new_cols = ['var_' + str(i) for i in np.arange(1, n+1)]
        return new_cols

    def generate(self, n_rows=100, n_basic_cols=10, loc=0,  noise_sigmas=None, basic_cost=1, seed=None, discretize_method='uniform', discretize_bins=10):
        X, y, costs = super().generate(n_rows=n_rows, n_basic_cols=n_basic_cols, loc=loc,  noise_sigmas=noise_sigmas, seed=seed, discretize_method=discretize_method, discretize_bins=discretize_bins)
        # Generate colnames
        if noise_sigmas is None:
            noise_sigmas_len = 0
        else:
            noise_sigmas_len = len(noise_sigmas)
        cols = self._generate_colnames(self.n_cols + self.n_cols*noise_sigmas_len)

        # Zip costs
        costs_dict = dict(zip(cols, costs))

        # Create final data frame
        X_df = pd.DataFrame(X, columns=cols)
        y_series = pd.Series(y)

        return X_df, y_series, costs_dict
