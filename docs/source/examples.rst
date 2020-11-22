========
Examples
========
Simple illustrative examples on how you can quickly start using the **bcselector** package.

Data Generation
---------------
First of all lets generate some artificial data, that we are going to use in feature selection. 
Bcselector provides two classes that let us generate data with costs:

- **MatrixGenerator** - generates data in *np.ndarray* and costs as *list*.
- **DataFrameGenerator** - generates data in *pd.DataFrame* and costs as *dict*.

Every method, uses the same algorithm, which is based on one main assumption, that mutual information of a feature and target variable is directly proportional to its cost.
Higher the cost, lower the noise.

1. Simulate :math:`p` independent random variables :math:`X_1,\ldots,X_p`, where :math:`X_i\sim N(0,1)`. We obtain :math:`p` variables :math:`X_i = \{x_1^{(i)},\ldots,x_n^{(i)}\}`, where :math:`n` is a sample size and :math:`c_i` is a cost for i-th variable. We assume that all costs are the same, i.e. :math:`c_i = c_1 = c_2 = \ldots = c_p = 1`.
2. For each observation :math:`(i)`, calculate the following term: :math:`\sigma_i = \frac{e^{\sum_{j=1}^p x_{i}^{(j)}}}{1+e^{\sum_{j=1}^p x_{i}^{(j)}}}.`
3. We generate target variable :math:`Y = \{y_1, \ldots, y_n\}`, where :math:`y_i` is generated from Bernoulli distribution with success probability :math:`\sigma_i`.
4. We generate :math:`p` noise random variables :math:`e_1,\ldots,e_p`, where :math:`e_i\sim N(0,\sigma)`.
5. We create new :math:`p` perturbed variables, each is generated as: :math:`X_i' := X_i + e_i`. Each variable :math:`X_i'` is assigned with cost equal to :math:`c_i' = \frac{1}{\sigma_i +1}`.
6. Steps :math:`4-5` are repeated for all values from list of standard deviations: :math:`noise\_sigmas = [\sigma_1, \ldots, \sigma_k]` 
7. At the end we obtain :math:`k*p` features. 

MatrixGenerator
~~~~~~~~~~~~~~~
.. code-block:: python

   from bcselector.data_generation import MatrixGenerator

   # Fix the seed for reproducibility.
   SEED = 42

   # Data generation arguments:
   # - data size, 
   # - cost of non-noised feature 
   # - sigma of noise for noised features.
   n_rows = 1000
   n_cols = 10
   noise_sigmas = [0.9,0.8,0.3,0.1]

   mg = MatrixGenerator()
   X, y, costs = mg.generate(
       n_rows=n_rows, 
       n_basic_cols=n_cols,
       noise_sigmas=noise_sigmas, 
       seed=SEED,
       discretize_method='uniform', 
       discretize_bins=10)

DataFrameGenerator
~~~~~~~~~~~~~~~~~~
.. code-block:: python

   from bcselector.data_generation import DataFrameGenerator

   # Fix the seed for reproducibility.
   SEED = 42

   # Data generation arguments:
   # - data size, 
   # - cost of non-noised feature,
   # - sigma of noise for noised features.
   n_rows = 1000
   n_cols = 10
   noise_sigmas = [0.9,0.8,0.3,0.1]

   dfg = DataFrameGenerator()
   X, y, costs = dfg.generate(
       n_rows=n_rows, 
       n_basic_cols=n_cols,
       noise_sigmas=noise_sigmas, 
       seed=SEED,
       discretize_method='uniform', 
       discretize_bins=10)

Feature Selection
-----------------
For this moment, just two methods of cost-sensitive feature selection methods are implemented:

- **FractionVariableSelector** - costs are compared to relation with target variable as difference.
- **DiffVariableSelector** - costs are compared to relation with target variable as fraction.

FractionVariableSelector
~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import roc_auc_score

   from bcselector.variable_selection import FractionVariableSelector
   from bcselector.data_generation import MatrixGenerator

   # Fix the seed for reproducibility.
   SEED = 42

   # Data generation arguments:
   # - data size, 
   # - cost of non-noised feature,
   # - sigma of noise for noised features.
   n_rows = 1000
   n_cols = 10
   noise_sigmas = [0.9,0.8,0.3,0.1]

   # Generate data
   mg = MatrixGenerator()
   X, y, costs = mg.generate(
       n_rows=n_rows, 
       n_basic_cols=n_cols,
       noise_sigmas=noise_sigmas, 
       seed=SEED,
       discretize_method='uniform', 
       discretize_bins=10)

   # Arguments for feature selection
   # - cost scaling parameter, 
   # - kwarg for j_criterion_func,
   # - model that is fitted on data.
   r = 1
   beta = 0.5
   model = LogisticRegression()

   # Feature selection
   fvs = FractionVariableSelector()
   fvs.fit(
        data=X,
        target_variable=y,
        costs=costs,
        r=r,
        j_criterion_func='cife',
        beta=beta)
   fvs.score(
        model=model, 
        scoring_function=roc_auc_score)
   fvs.plot_scores(
        compare_no_cost_method=True, 
        model=model, 
        annotate=True)

DiffVariableSelector
~~~~~~~~~~~~~~~~~~~~~~~~
.. code-block:: python

   from sklearn.linear_model import LogisticRegression
   from sklearn.metrics import roc_auc_score

   from bcselector.variable_selection import DiffVariableSelector
   from bcselector.data_generation import MatrixGenerator
   

   # Fix the seed for reproducibility.
   SEED = 42

   # Data generation arguments:
   # - data size, 
   # - cost of non-noised feature,
   # - sigma of noise for noised features.
   n_rows = 1000
   n_cols = 10
   noise_sigmas = [0.9,0.8,0.3,0.1]

   # Generate data
   mg = MatrixGenerator()
   X, y, costs = mg.generate(
       n_rows=n_rows, 
       n_basic_cols=n_cols,
       noise_sigmas=noise_sigmas, 
       seed=SEED,
       discretize_method='uniform', 
       discretize_bins=10)

   # Arguments for feature selection
   # - cost scaling parameter, 
   # - model that is fitted on data.
   lamb = 1
   beta = 0.5
   model = LogisticRegression()

   # Feature selection
   dvs = DiffVariableSelector()
   dvs.fit(
        data=X,
        target_variable=y,
        costs=costs,
        lamb=lamb,
        j_criterion_func='jmi')
   dvs.score(
        model=model, 
        scoring_function=roc_auc_score)
   dvs.plot_scores(
        compare_no_cost_method=True, 
        model=model, 
        annotate=True)
