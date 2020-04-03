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

class MatrixGenerator(_BasicDataGenerator):
    def __init__(self):
        super().__init__()
    
    def _generate_basic_dataset(self, loc = 0, scale = 1):
        np.random.seed(seed = self.seed)
        X = np.random.normal(loc = loc, scale = scale, size = (self.n_rows,self.n_cols))
        y = np.random.binomial(1, np.exp(X.sum(axis=1))/(1+np.exp(X.sum(axis=1))))
        return X,y

    def _generate_noise(self, sigma_min, sigma_max, loc = 0):
        np.random.seed(seed = self.seed)
        noise_sigmas = np.random.uniform(sigma_min, sigma_max, size = self.n_cols)
        noise = np.random.normal(loc = loc, scale = noise_sigmas, size = (self.n_rows,self.n_cols))
        return noise, noise_sigmas

    def generate(self,n_rows = 100, n_cols = 10, loc = 0,  noise_sigma_range = (0,1), seed = None, round_level = None):
        assert isinstance(n_rows, int), "Argument `n_rows` must be int."
        assert isinstance(n_cols, int), "Argument `n_cols` must be int."
        assert isinstance(loc, int), "Argument `loc` must be int or float."
        assert isinstance(noise_sigma_range, tuple) and len(noise_sigma_range) == 2, "Argument `noise_sigma_man_std` must be tuple of length 2."

        super().generate(n_rows, n_cols, seed)
        np.random.seed(seed = self.seed)

        self.loc = 0
        self.noise_sigma_min = noise_sigma_range[0]
        self.noise_sigma_max = noise_sigma_range[1]

        # Generate basic dataset
        X,y = self._generate_basic_dataset(loc=0, scale=1)
        # Generate noise
        noise, noise_sigmas = self._generate_noise(sigma_min=self.noise_sigma_min, sigma_max=self.noise_sigma_max)
        X_transformed = X + noise

        # Calculate costs
        costs = 1/noise_sigmas

        # Round output if selected
        if round_level:
            X_transformed = X_transformed.round(round_level)
        
        return X_transformed, y, list(costs)

class DataFrameGenerator(MatrixGenerator):
    def __init__(self):
        super().__init__()

    def _generate_colnames(self, n):
        new_cols = ['var_' + str(i) for i in np.arange(1,n+1)]
        return new_cols

    def generate(self,n_rows = 100, n_cols = 10, loc = 0,  noise_sigma_range = (0,1), seed = None, round_level = None):
        X,y,costs = super().generate(n_rows = n_rows, n_cols = n_cols, loc = loc,  noise_sigma_range = noise_sigma_range, seed = seed, round_level = round_level)
        # Generate colnames
        cols = self._generate_colnames(self.n_cols)

        # Zip costs
        costs_dict = dict(zip(cols,costs))

        # Create final data frame
        X_df = pd.DataFrame(X, columns = cols)
        y_series = pd.Series(y)

        return X_df, y_series, costs_dict