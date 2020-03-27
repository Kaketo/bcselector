import unittest
import numpy as np

from bcselector.information_theory.basic_approximations import entropy, conditional_entropy, mutual_information, conditional_mutual_information

class TestEntropy(unittest.TestCase):
    def test_empty_input(self):
        input = []
        with self.assertRaises(AssertionError): entropy(input)

    def test_one_number_input(self):
        self.assertEqual(entropy([1]),0.0)

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
    def test_empty_input(self):
        input = []
        condition = []
        with self.assertRaises(AssertionError): conditional_entropy(input, condition)

    def test_one_number_input(self):
        self.assertEqual(conditional_entropy([1],[0]),0.0)

    def test_different_input_sizes(self):
        input = [0,1]
        condition = [1,2,3]
        with self.assertRaises(AssertionError): conditional_entropy(input, condition)

    def test_list_input(self):
        list_input = [1,2,3,5,432,42,31234,342,34]
        condition = [1,1,1,1,0,0,0,0,0]
        self.assertIsInstance(conditional_entropy(list_input,condition),float)
    
    def test_nparray_input(self):
        np_input = np.array([1,2,3,5,432,42,31234,342,34])
        condition = np.array([1,1,1,1,0,0,0,0,0])
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

class TestMutualInformation(unittest.TestCase):
    def test_empty_input(self):
        vector_1 = []
        vector_2 = []
        with self.assertRaises(AssertionError): mutual_information(vector_1, vector_2)
        
    def test_one_number_input(self):
        self.assertEqual(mutual_information([1],[0]),0.0)

    def test_different_input_sizes(self):
        vector_1 = [2,0]
        vector_2 = [1,2,3]
        with self.assertRaises(AssertionError): mutual_information(vector_1, vector_2)
        with self.assertRaises(AssertionError): mutual_information(vector_2, vector_1)

    def test_list_input(self):
        vector_1 = [1,2,3,5,432,42,31234,342,34]
        vector_2 = [1,1,1,1,0,0,0,0,0]
        self.assertIsInstance(mutual_information(vector_1,vector_2),float)
    
    def test_nparray_input(self):
        vector_1 = np.array([1,2,3,5,432,42,31234,342,34])
        vector_2 = np.array([1,1,1,1,0,0,0,0,0])
        self.assertIsInstance(mutual_information(vector_1, vector_2),float)

    def test_theoretical_output(self):
        """
        proper_value is calculated using R function infotheo::mutinformation(method="emp")
        """
        input_1 = [9,8,7,6,5,4,3,2,9]
        input_2 = [1,1,1,1,0,0,0,0,0]
        proper_value_1 = 0.5329289
        self.assertAlmostEqual(mutual_information(input_1, input_2),proper_value_1,places=3)

        input_3 = [0,0,0,0,1,0,0,0,0]
        input_4 = [1,1,1,1,0,0,0,0,0]
        proper_value_2 = 0.07083075
        self.assertAlmostEqual(mutual_information(input_3, input_4),proper_value_2,places=3) 

    def test_the_same_vectors(self):
        input_1 = [9,8,7,6,5,4,3,2,9]
        input_2 = [9,8,7,6,5,4,3,2,9]

        self.assertEqual(mutual_information(input_1,input_2), entropy(input_1))

    def test_commutative_property(self):
        input_1 = [9,8,7,6,5,4,3,2,9]
        input_2 = [1,1,1,1,0,0,0,0,0]
        
        self.assertEqual(mutual_information(input_1, input_2), mutual_information(input_2, input_1))

class TestConditionalMutualInformation(unittest.TestCase):
    def test_empty_input(self):
        vector_1 = []
        vector_2 = []
        condition = []
        with self.assertRaises(AssertionError): conditional_mutual_information(vector_1, vector_2,condition)
        
    def test_one_number_input(self):
        self.assertEqual(conditional_mutual_information([1],[0],[1]),0.0)

    def test_different_input_sizes(self):
        vector_1 = [2,0]
        vector_2 = [1,2,3]
        condition = [0,0,0,1]
        with self.assertRaises(AssertionError): conditional_mutual_information(vector_1, vector_2, condition)
        with self.assertRaises(AssertionError): conditional_mutual_information(vector_2, vector_1, condition)

    def test_list_input(self):
        vector_1 = [1,2,3,5,432,42,31234,342,34]
        vector_2 = [1,1,1,1,0,0,0,0,0]
        condition = [0,0,0,0,1,0,0,0,0]
        self.assertIsInstance(conditional_mutual_information(vector_1,vector_2,condition),float)
    
    def test_nparray_input(self):
        vector_1 = np.array([1,2,3,5,432,42,31234,342,34])
        vector_2 = np.array([1,1,1,1,0,0,0,0,0])
        condition = np.array([0,0,0,0,1,0,0,0,0])
        self.assertIsInstance(conditional_mutual_information(vector_1, vector_2,condition),float)

    def test_commutative_property(self):
        input_1 = [9,8,7,6,5,4,3,2,9]
        input_2 = [1,1,1,1,0,0,0,0,0]
        condition = np.array([0,0,0,0,1,0,0,0,0])
        
        self.assertAlmostEqual(conditional_mutual_information(input_1, input_2, condition), conditional_mutual_information(input_2, input_1, condition), places=5)

if __name__ == '__main__':
    unittest.main()