import unittest
import numpy as np

from bcselector.filter_methods.no_cost_based_filter_methods import no_cost_find_best_feature
from bcselector.information_theory.j_criterion_approximations import mim, mifs, mrmr, jmi, cife


class TestNoCostMethod(unittest.TestCase):
    def test_simple_input_mim(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]

        selected_feature, criterion_value, cost = no_cost_find_best_feature(
            j_criterion_func=mim, 
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
        beta = 10

        selected_feature, criterion_value, cost = no_cost_find_best_feature(
                                    j_criterion_func=mifs, 
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs,
                                    prev_variables_index=prev_variables_index,
                                    beta=beta)
        self.assertIsInstance(selected_feature,int)
        self.assertIsInstance(criterion_value, float)
        self.assertIsInstance(cost, float)  
        
    def test_simple_input_mifs_no_beta_provided(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        prev_variables_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]

        with self.assertWarns(Warning): no_cost_find_best_feature(
                                            j_criterion_func=mifs, 
                                            data=integer_matrix, 
                                            target_variable=diverse_target,
                                            possible_variables_index=candidates_index,
                                            costs=costs,
                                            prev_variables_index=prev_variables_index)   

    def test_simple_input_mrmr(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        prev_variable_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        selected_feature, criterion_value, cost = no_cost_find_best_feature(
                                    j_criterion_func=mrmr, 
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

        selected_feature, criterion_value, cost = no_cost_find_best_feature(
                                    j_criterion_func=jmi, 
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
        beta = 10

        selected_feature, criterion_value, cost = no_cost_find_best_feature(
                                    j_criterion_func=cife, 
                                    data=integer_matrix, 
                                    target_variable=diverse_target,
                                    possible_variables_index=candidates_index,
                                    costs=costs, 
                                    prev_variables_index=prev_variable_index,
                                    beta=beta)
        self.assertIsInstance(selected_feature,int)
        self.assertIsInstance(criterion_value, float)
        self.assertIsInstance(cost, float) 

    def test_simple_input_cife_no_beta_provided(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        prev_variables_index = [3,4,5]
        candidates_index = [0,1,2,6,7,8,9]
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]

        with self.assertWarns(Warning): no_cost_find_best_feature(
                                            j_criterion_func=cife, 
                                            data=integer_matrix, 
                                            target_variable=diverse_target,
                                            possible_variables_index=candidates_index,
                                            costs=costs,
                                            prev_variables_index=prev_variables_index)   

if __name__ == '__main__':
    unittest.main()