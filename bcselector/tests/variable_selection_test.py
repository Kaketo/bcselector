import unittest
import numpy as np
import pandas as pd

from bcselector.variable_selection import DiffVariableSelector

class TestDiffVariableSelector(unittest.TestCase):
    def test_numpy_input(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        lamb = 1

        dvs = DiffVariableSelector()
        dvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                lamb=lamb,
                j_criterion_func='mim')

        self.assertIsInstance(dvs.variables_selected_order, list)
        self.assertEqual(len(dvs.variables_selected_order), len(costs))

    def test_pandas_input(self):
        integer_matrix = pd.DataFrame(np.random.randint(0,10,(100,3)), columns=['AA','BB','CC'])
        diverse_target = np.random.randint(0,2,(100))
        costs = {'AA': 10, 'BB': 1, 'CC': 1.5}
        lamb = 1

        dvs = DiffVariableSelector()
        dvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                lamb=lamb,
                j_criterion_func='mim')

        self.assertIsInstance(dvs.variables_selected_order, list)
        self.assertEqual(len(dvs.variables_selected_order), len(costs))

    def test_theoretical_output(self):
        integer_matrix = np.array([[0,1,0],[0,1,0],[0,1,2],[0,1,3],[1,1,5]])
        diverse_target = np.array([0,0,0,0,1])
        costs = [1,1,1]
        lamb = 1

        dvs = DiffVariableSelector()
        dvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                lamb=lamb,
                j_criterion_func='mim')
        
        self.assertEqual(dvs.variables_selected_order[0], 2)


