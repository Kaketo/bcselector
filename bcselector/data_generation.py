import numpy as np
import pandas as pd
import sys

class _BasicDataGenerator():
    def __init__(self):
        self.n_rows = None
        self.n_cols = None
        self.seed = None
        
    def generate(self,n_rows = 100, n_cols = 10, seed = None):
        self.n_rows = n_rows
        self.n_cols = n_cols
        if seed is None:
            self.seed = np.random.randint(0, 420000)
        else:
            self.seed = seed
        np.random.seed(seed = self.seed)

class MatrixGenerator(_BasicDataGenerator):
    def __init__(self):
        super().__init__()

    def generate(self,n_rows = 100, n_cols = 10, noise_sigma_man_std = (0,1)):
        assert isinstance(n_rows, int), "Argument `n_rows` must be int."
        assert isinstance(n_cols, int), "Argument `n_cols` must be int."
        assert isinstance(noise_sigma_man_std, tuple) and len(noise_sigma_man_std) == 2, "Argument `noise_sigma_man_std` must be tuple of length 2."

        super().generate(n_rows, n_cols, self.seed)
        
        self.noise_sigma_mean = noise_sigma_man_std[0]
        self.noise_sigma_std = noise_sigma_man_std[1]

        # Generate basic dataset
        X = np.random.normal(loc = 0, scale = 1, size = (self.n_rows,self.n_cols))
        y = 1 - np.random.binomial(1, np.exp(X.sum(axis=1))/(1+np.exp(X.sum(axis=1))))
    
        # Generate noise
        noise_sigmas = abs(np.random.normal(loc=self.noise_sigma_mean, scale=self.noise_sigma_std, size = X.shape[1]))
        noise = np.random.normal(loc = 0, scale = noise_sigmas, size = (n_rows,n_cols))
        X_transformed = X + noise

        # Calculate costs
        costs = 1/noise_sigmas*self.noise_sigma_std
        
        return X_transformed, y, list(costs)

class DataFrameGenerator(_BasicDataGenerator):
    def __init__(self):
        super().__init__()

    def _generate_colnames(self, n):
        new_cols = ['var_' + str(i) for i in np.arange(1,n+1)]
        return new_cols

    def generate(self,n_rows = 100, n_cols = 10, noise_sigma_man_std = (0,1)):
        assert isinstance(n_rows, int), "Argument `n_rows` must be int."
        assert isinstance(n_cols, int), "Argument `n_cols` must be int."
        assert isinstance(self.seed, int), "Argument `seed` must be int."
        assert isinstance(noise_sigma_man_std, tuple) and len(noise_sigma_man_std) == 2, "Argument `noise_sigma_man_std` must be tuple of length 2."

        self.noise_sigma_mean = noise_sigma_man_std[0]
        self.noise_sigma_std = noise_sigma_man_std[1]

        super().generate(n_rows, n_cols, self.seed)
        # Generate basic dataset
        X = np.random.normal(loc = 0, scale = 1, size = (self.n_rows,self.n_cols))
        y = 1 - np.random.binomial(1, np.exp(X.sum(axis=1))/(1+np.exp(X.sum(axis=1))))
        
        # Generate noise
        noise_sigmas = abs(np.random.normal(loc=self.noise_sigma_mean, scale=self.noise_sigma_std, size = X.shape[1]))
        noise = np.random.normal(loc = 0, scale = noise_sigmas, size = (n_rows,n_cols))
        X_transformed = X + noise

        # Generate colnames
        cols = self._generate_colnames(self.n_cols)

        # Calculate costs
        costs = 1/noise_sigmas*self.noise_sigma_std
        costs_dict = dict(zip(cols,costs))   
        
        # Create DF
        X_df = pd.DataFrame(X_transformed, columns=cols)
        y_series = pd.Series(y)

        return X_df, y_series, costs_dict