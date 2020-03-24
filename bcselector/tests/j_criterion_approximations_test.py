import unittest
import numpy as np
np.random.seed = 42

from bcselector.information_theory.j_criterion_approximations import mim, mifs, mrmr, jmi, cife
from bcselector.information_theory.basic_approximations import mutual_information, conditional_mutual_information


class TestMIM(unittest.TestCase):
    # def __init__(self):
        
    #     self.empty_matrix = np.array([[],[]])
    #     self.zeros_matrix = np.zeros((100,50))
    #     self.integer_matrix = np.random.randint(0,10,(100,50))
    #     self.float_matrix = np.random.normal(0,1,(100,50))

    #     self.diverse_target = np.random.randint(0,10,(100))
    #     self.non_diverse_target = np.random.binomial(1,0.1,100)
    #     self.one_class_target = np.repeat(1,100)
    
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