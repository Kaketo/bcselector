import unittest
import numpy as np
np.random.seed = 42

from bcselector.information_theory.j_criterion_approximations import mim, mifs, mrmr, jmi, cife


class TestMIM(unittest.TestCase):
    def __init__(self):
        
        self.empty_matrix = np.array([[],[]])
        self.zeros_matrix = np.zeros((100,50))
        self.integer_matrix = np.random.randint(0,10,(100,50))
        self.float_matrix = np.random.normal(0,1,(100,50))

        self.diverse_target = np.random.randint(0,10,(100))
        self.non_diverse_target = np.random.binomial(1,0.1,100)
        self.one_class_target = np.repeat(1,100)
    
    def test_empty_imput(self):
        with self.assertRaises(AssertionError): mim(self.empty_matrix, self.diverse_target, 0)

