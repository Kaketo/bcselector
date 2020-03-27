import unittest
import numpy as np

from bcselector.information_theory.j_criterion_approximations import mim, mifs, mrmr, jmi, cife
from bcselector.information_theory.basic_approximations import mutual_information, conditional_mutual_information


class TestMIM(unittest.TestCase):
    def test_empty_imput(self):
        empty_matrix = np.array([[],[]])
        diverse_target = np.random.randint(0,10,(100))
        with self.assertRaises(AssertionError): mim(empty_matrix, diverse_target, 0)

    def test_too_high_candidate_index(self):
        zeros_matrix = np.zeros((100,50))
        diverse_target = np.random.randint(0,10,(100))
        too_high_index = 51
        with self.assertRaises(AssertionError): mim(zeros_matrix, diverse_target, too_high_index)        

    def test_theoretical_value(self):
        integer_matrix = np.random.randint(0,10,(100,50))
        diverse_target = np.random.randint(0,10,(100))
        candidate_index = 1
        self.assertAlmostEqual(mim(integer_matrix, diverse_target, candidate_index), mutual_information(diverse_target,integer_matrix[:,candidate_index]), places = 5)

    def test_one_class_target(self):
        integer_matrix = np.random.randint(0,10,(100,50))
        one_class_target = np.repeat(1,100)
        candidate_index = 1
        self.assertAlmostEqual(mim(integer_matrix, one_class_target, candidate_index), 0.0, places = 5)

class TestMIFS(unittest.TestCase):
    def test_empty_imput(self):
        empty_matrix = np.array([[],[]])
        diverse_target = np.random.randint(0,10,(100))
        prev_variable_index = [3,4,5]
        candidate_index = 1
        with self.assertRaises(AssertionError): mifs(empty_matrix, diverse_target, prev_variable_index, candidate_index, beta = 10)

    def test_no_beta_provided(self):
        zeros_matrix = np.zeros((100,50))
        diverse_target = np.random.randint(0,10,(100))
        prev_variable_index = [3,4,5]
        candidate_index = 1
        with self.assertWarns(Warning): mifs(zeros_matrix, diverse_target, prev_variable_index, candidate_index)

    def test_too_high_candidate_index(self):
        zeros_matrix = np.zeros((100,50))
        diverse_target = np.random.randint(0,10,(100))
        prev_variable_index = [3,4,5]
        too_high_index = 51
        beta = 2
        with self.assertRaises(AssertionError): mifs(zeros_matrix, diverse_target, prev_variable_index, too_high_index, beta=beta)    

    def test_theoretical_value(self):
        integer_matrix = np.random.randint(0,10,(100,50))
        diverse_target = np.random.randint(0,10,(100))
        prev_variable_index = [3,4,5]
        candidate_index = 1
        beta_1 = 1
        beta_2 = 10
        self.assertGreater(mifs(integer_matrix,diverse_target,prev_variable_index,candidate_index,beta = beta_1), 
                            mifs(integer_matrix,diverse_target,prev_variable_index,candidate_index,beta = beta_2))

if __name__ == '__main__':
    unittest.main()