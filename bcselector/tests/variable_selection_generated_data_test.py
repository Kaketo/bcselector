import unittest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bcselector.variable_selection import FractionVariableSelector
from bcselector.data_generation import MatrixGenerator


class TestMatrixGenerator(unittest.TestCase):
    def test_cife(self):
        # Given
        n_cols = 5
        n_rows = 1000
        model = LogisticRegression()

        # When
        mg = MatrixGenerator()
        X, y, costs = mg.generate(n_rows=n_rows, n_basic_cols=n_cols, noise_sigmas=[0.1, 0.5], seed=2)
        r = 1
        beta = 0.5

        dvs = FractionVariableSelector()
        dvs.fit(data=X,
                target_variable=y,
                costs=costs,
                r=r,
                j_criterion_func='cife',
                beta=beta)
        dvs.score(model=model, scoring_function=roc_auc_score)
        dvs.plot_scores(compare_no_cost_method=True, model=model, annotate=True)

        # Then
        self.assertIsInstance(dvs.variables_selected_order, list)
        self.assertEqual(len(dvs.variables_selected_order), len(costs))
        self.assertAlmostEqual(sum(costs), sum(dvs.cost_variables_selected_order))

    def test_regard_to_cost_is_better_cife(self):
        # Given
        n_cols = 3
        n_rows = 1000
        model = LogisticRegression()
        sigmas = [1, 10, 100]

        # When
        mg = MatrixGenerator()
        X, y, costs = mg.generate(n_rows=n_rows, n_basic_cols=n_cols, basic_cost=1, noise_sigmas=sigmas, seed=42)
        r = 0.8

        fvs = FractionVariableSelector()
        fvs.fit(data=X,
                target_variable=y,
                costs=costs,
                r=r,
                j_criterion_func='cife',
                beta=0.05)
        fvs.score(model=model, scoring_function=roc_auc_score)
        fvs.plot_scores(compare_no_cost_method=True, model=model)

        def find_nearest_idx(list, value):
            array = np.asarray(list)
            idx = (np.abs(array - value)).argmin()
            return idx

        when_better = []
        for i in range(fvs.data.shape[1]):
            idx_1_no_cost = i
            idx_1_cost = find_nearest_idx(fvs.total_costs, fvs.no_cost_total_costs[idx_1_no_cost])
            if fvs.total_scores[idx_1_cost] > fvs.no_cost_total_scores[idx_1_no_cost]:
                when_better.append(True)
            else:
                when_better.append(False)

        # Then
        self.assertTrue(sum(when_better)/len(when_better) >= 0.5)

    def test_regard_to_cost_is_better(self):
        # Given
        n_cols = 3
        n_rows = 1000
        model = LogisticRegression()
        sigmas = [1, 10, 100]

        # When
        mg = MatrixGenerator()
        X, y, costs = mg.generate(n_rows=n_rows, n_basic_cols=n_cols, basic_cost=1, noise_sigmas=sigmas, seed=42)
        r = 0.8

        fvs = FractionVariableSelector()
        fvs.fit(data=X,
                target_variable=y,
                costs=costs,
                r=r,
                j_criterion_func='mim')
        fvs.score(model=model, scoring_function=roc_auc_score)
        fvs.plot_scores(compare_no_cost_method=True, model=model)

        def find_nearest_idx(list, value):
            array = np.asarray(list)
            idx = (np.abs(array - value)).argmin()
            return idx

        when_better = []
        for i in range(fvs.data.shape[1]):
            idx_1_no_cost = i
            idx_1_cost = find_nearest_idx(fvs.total_costs, fvs.no_cost_total_costs[idx_1_no_cost])
            if fvs.total_scores[idx_1_cost] > fvs.no_cost_total_scores[idx_1_no_cost]:
                when_better.append(True)
            else:
                when_better.append(False)

        # Then
        self.assertTrue(sum(when_better)/len(when_better) >= 0.5)


if __name__ == '__main__':
    unittest.main()
