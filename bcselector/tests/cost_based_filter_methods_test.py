import unittest
import numpy as np

from bcselector.filter_methods.cost_based_filter_methods import fraction_find_best_feature, difference_find_best_feature
from bcselector.information_theory.j_criterion_approximations import mim, mifs, mrmr, jmi, cife


class TestFractionMethod(unittest.TestCase):
    def test_simple_input_mim(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        r = 1
        selected_feature, criterion_value, cost = fraction_find_best_feature(
            j_criterion_func=mim, 
            r=r,
            data=integer_matrix, 
            target_variable=diverse_target,
            possible_variables_index=candidates_index,
            costs=costs)
        self.assertIsInstance(selected_feature,int)
        self.assertIsInstance(criterion_value, float)
        self.assertIsInstance(cost, float)

    def test_simple_input_mifs(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        prev_variables_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        r = 1
        beta = 1
        selected_feature, criterion_value, cost = fraction_find_best_feature(
                                    j_criterion_func=mifs, 
                                    r=r,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs,
                                    prev_variables_index=prev_variables_index,
                                    beta=beta)
        self.assertIsInstance(selected_feature,int)
        self.assertIsInstance(criterion_value, float)
        self.assertIsInstance(cost, float)                                    

    def test_simple_input_mrmr(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        prev_variable_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        r = 1
        selected_feature, criterion_value, cost = fraction_find_best_feature(
                                    j_criterion_func=mrmr, 
                                    r=r,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs, 
                                    prev_variables_index=prev_variable_index)
        self.assertIsInstance(selected_feature,int)
        self.assertIsInstance(criterion_value, float)
        self.assertIsInstance(cost, float) 

    def test_simple_input_jmi(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        prev_variable_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        r = 1
        selected_feature, criterion_value, cost = fraction_find_best_feature(
                                    j_criterion_func=jmi, 
                                    r=r,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs, 
                                    prev_variables_index=prev_variable_index)
        self.assertIsInstance(selected_feature,int)
        self.assertIsInstance(criterion_value, float)
        self.assertIsInstance(cost, float) 

    def test_simple_input_cife(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        prev_variable_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        r = 1
        beta=1
        selected_feature, criterion_value, cost = fraction_find_best_feature(
                                    j_criterion_func=cife, 
                                    r=r,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs, 
                                    prev_variables_index=prev_variable_index,
                                    beta=beta)
        self.assertIsInstance(selected_feature,int)
        self.assertIsInstance(criterion_value, float)
        self.assertIsInstance(cost, float) 

    def test_different_beta_parameter_mifs(self):
        integer_matrix = np.random.randint(0,10,(10,10))
        diverse_target = np.random.randint(0,10,(10))
        prev_variables_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        r = 1
        beta_1 = 1
        beta_2 = 10000
        _, criterion_value_1, _ = fraction_find_best_feature(
                                    j_criterion_func=mifs, 
                                    r=r,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs,
                                    prev_variables_index=prev_variables_index,
                                    beta=beta_1)
        _, criterion_value_2, _ = fraction_find_best_feature(
                                    j_criterion_func=mifs, 
                                    r=r,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs,
                                    prev_variables_index=prev_variables_index,
                                    beta=beta_2)
        self.assertNotEqual(criterion_value_1,criterion_value_2)
                 
    def test_different_beta_parameter_cife(self):
        integer_matrix = np.random.randint(0,10,(10,10))
        diverse_target = np.random.randint(0,10,(10))
        prev_variables_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        r = 1
        beta_1 = 1
        beta_2 = 10000
        _, criterion_value_1, _ = fraction_find_best_feature(
                                    j_criterion_func=cife, 
                                    r=r,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs,
                                    prev_variables_index=prev_variables_index,
                                    beta=beta_1)
        _, criterion_value_2, _ = fraction_find_best_feature(
                                    j_criterion_func=cife, 
                                    r=r,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs,
                                    prev_variables_index=prev_variables_index,
                                    beta=beta_2)
        self.assertNotEqual(criterion_value_1,criterion_value_2)


class TestDifferenceMethod(unittest.TestCase):
    def test_simple_input_mim(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        # prev_variable_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        lamb = 1
        selected_feature, criterion_value, cost = difference_find_best_feature(
            j_criterion_func=mim, 
            lamb = lamb,
            data=integer_matrix, 
            target_variable=diverse_target,
            possible_variables_index=candidates_index,
            costs=costs)
        self.assertIsInstance(selected_feature,int)
        self.assertIsInstance(criterion_value, float)
        self.assertIsInstance(cost, float)

    def test_simple_input_mifs(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        prev_variables_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        lamb = 1
        beta = 1
        selected_feature, criterion_value, cost = difference_find_best_feature(
                                    j_criterion_func=mifs, 
                                    lamb = lamb,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs,
                                    prev_variables_index=prev_variables_index,
                                    beta=beta)
        self.assertIsInstance(selected_feature,int)
        self.assertIsInstance(criterion_value, float)
        self.assertIsInstance(cost, float)                                    

    def test_simple_input_mrmr(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        prev_variable_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        lamb = 1
        selected_feature, criterion_value, cost = difference_find_best_feature(
                                    j_criterion_func=mrmr, 
                                    lamb = lamb,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs, 
                                    prev_variables_index=prev_variable_index)
        self.assertIsInstance(selected_feature,int)
        self.assertIsInstance(criterion_value, float)
        self.assertIsInstance(cost, float) 

    def test_simple_input_jmi(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        prev_variable_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        lamb = 1
        selected_feature, criterion_value, cost = difference_find_best_feature(
                                    j_criterion_func=jmi, 
                                    lamb = lamb,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs, 
                                    prev_variables_index=prev_variable_index)
        self.assertIsInstance(selected_feature,int)
        self.assertIsInstance(criterion_value, float)
        self.assertIsInstance(cost, float) 

    def test_simple_input_cife(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        prev_variable_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        lamb = 1
        beta=1
        selected_feature, criterion_value, cost = difference_find_best_feature(
                                    j_criterion_func=cife, 
                                    lamb = lamb,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs, 
                                    prev_variables_index=prev_variable_index,
                                    beta=beta)
        self.assertIsInstance(selected_feature,int)
        self.assertIsInstance(criterion_value, float)
        self.assertIsInstance(cost, float) 

    def test_different_beta_parameter_mifs(self):
        integer_matrix = np.random.randint(0,10,(10,10))
        diverse_target = np.random.randint(0,10,(10))
        prev_variables_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        lamb = 1
        beta_1 = 1
        beta_2 = 10000
        _, criterion_value_1, _ = difference_find_best_feature(
                                    j_criterion_func=mifs, 
                                    lamb=lamb,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs,
                                    prev_variables_index=prev_variables_index,
                                    beta=beta_1)
        _, criterion_value_2, _ = difference_find_best_feature(
                                    j_criterion_func=mifs, 
                                    lamb=lamb,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs,
                                    prev_variables_index=prev_variables_index,
                                    beta=beta_2)
        self.assertNotEqual(criterion_value_1,criterion_value_2)
                 
    def test_different_beta_parameter_cife(self):
        integer_matrix = np.random.randint(0,10,(10,10))
        diverse_target = np.random.randint(0,10,(10))
        prev_variables_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        lamb = 1
        beta_1 = 1
        beta_2 = 10000
        _, criterion_value_1, _ = difference_find_best_feature(
                                    j_criterion_func=cife, 
                                    lamb=lamb,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs,
                                    prev_variables_index=prev_variables_index,
                                    beta=beta_1)
        _, criterion_value_2, _ = difference_find_best_feature(
                                    j_criterion_func=cife, 
                                    lamb=lamb,
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs,
                                    prev_variables_index=prev_variables_index,
                                    beta=beta_2)
        self.assertNotEqual(criterion_value_1,criterion_value_2)

if __name__ == '__main__':
    unittest.main()