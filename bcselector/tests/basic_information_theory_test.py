import unittest
import numpy as np

from bcselector.information_theory.basic_approximations import entropy, conditional_entropy, mutual_information, conditional_mutual_information

class TestEntropy(unittest.TestCase):
    def test_list_input(self):
        list_input = [1,2,3,5,432,42,31234,342,34]
        self.assertIsInstance(entropy(list_input),float)
    
    def test_nparray_input(self):
        np_input = np.array([1,2,3,5,432,42,31234,342,34])
        self.assertIsInstance(entropy(np_input),float)

    def test_theoretical_output(self):
        """
        proper_value is calculated using R function infotheo::entropy(method="emp")
        """
        input_1 = [9,8,7,6,5,4,3,2,9]
        proper_value_1 = 2.043192
        self.assertAlmostEqual(entropy(input_1),proper_value_1,places=3)

        input_2 = [0,0,0,0,1,0,0,0,0]
        proper_value_2 = 0.3488321
        self.assertAlmostEqual(entropy(input_2),proper_value_2,places=3)

        # Entropy must be higher in more diverse vectors
        self.assertGreater(entropy(input_1), entropy(input_2))

class TestConditionalEntropy(unittest.TestCase):
    def test_list_input(self):
        list_input = [1,2,3,5,432,42,31234,342,34]
        condition = [1,1,1,1,0,0,0,0,0]
        self.assertIsInstance(conditional_entropy(list_input,condition),float)
    
    def test_nparray_input(self):
        np_input = np.array([1,2,3,5,432,42,31234,342,34])
        condition = [1,1,1,1,0,0,0,0,0]
        self.assertIsInstance(conditional_entropy(np_input, condition),float)

    def test_theoretical_output(self):
        """
        proper_value is calculated using R function infotheo::condentropy(method="emp")
        """
        input_1 = [9,8,7,6,5,4,3,2,9]
        cond_1 = [1,1,1,1,0,0,0,0,0]
        proper_value_1 = 1.510263
        self.assertAlmostEqual(conditional_entropy(input_1, cond_1),proper_value_1,places=3)

        input_2 = [0,0,0,0,1,0,0,0,0]
        cond_2 = [1,1,1,1,0,0,0,0,0]
        proper_value_2 = 0.2780013
        self.assertAlmostEqual(conditional_entropy(input_2, cond_2),proper_value_2,places=3)

        # Entropy must be higher in more diverse vectors
        self.assertGreater(conditional_entropy(input_1, cond_1), conditional_entropy(input_2, cond_2))



    