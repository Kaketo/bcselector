import unittest
import numpy as np
import pandas as pd

from bcselector.variable_selection import DiffVariableSelector, FractionVariableSelector, NoCostVariableSelector
from bcselector.data_generation import MatrixGenerator, DataFrameGenerator
from bcselector.filter_methods.cost_based_filter_methods import fraction_find_best_feature
from bcselector.information_theory.j_criterion_approximations import mim, mifs
from bcselector.information_theory.basic_approximations import mutual_information


class TestFVS_r_paramter(unittest.TestCase):
    def test_criterion_filter_values(self):
        target = [1,0,1,1,0,1,0,1]
        a = [1,0,0,1,0,1,0,1]
        b = [1,0,0,1,1,1,0,1]
        c = [1,1,1,1,1,1,1,0]
        d = [1,0,0,1,1,0,0,1]

        X = np.array([a,b,c,d]).transpose()
        y = np.array(target).transpose()
        costs = [1,0.5,0.25,0.1]
        normalized_costs = list((np.array(costs) - min(costs) + 0.0001)/(max(costs)-min(costs)+0.0001))

        # MIM
        r = 0
        feature_index, filter_value, criterion_value, cost = fraction_find_best_feature(j_criterion_func=mim, r=r, data=X, target_variable=y, possible_variables_index = [0,1,2,3], costs=costs, normalized_costs=normalized_costs)
        self.assertAlmostEqual(mutual_information(y, X[:,feature_index]), criterion_value)

        r = 1.2
        feature_index, filter_value, criterion_value, cost = fraction_find_best_feature(j_criterion_func=mim, r=r, data=X, target_variable=y, possible_variables_index = [0,1,2,3], costs=costs, normalized_costs=normalized_costs)
        self.assertAlmostEqual(mutual_information(y, X[:,feature_index]), criterion_value)
        self.assertAlmostEqual(mutual_information(y, X[:,feature_index])/normalized_costs[feature_index]**r, filter_value)

        # MIFS
        r = 0
        feature_index, filter_value, criterion_value, cost = fraction_find_best_feature(j_criterion_func=mifs, 
                                                                                        r=r, 
                                                                                        data=X, 
                                                                                        target_variable=y, 
                                                                                        possible_variables_index = [1,2],
                                                                                        costs=costs,
                                                                                        normalized_costs=normalized_costs,
                                                                                        prev_variables_index = [0,3])
        mifs_value = mutual_information(y, X[:,feature_index]) - mutual_information(X[:,feature_index],X[:,0])-mutual_information(X[:,feature_index],X[:,3])                                                        
        self.assertAlmostEqual(mifs_value, criterion_value)

        r = 1
        feature_index, filter_value, criterion_value, cost = fraction_find_best_feature(j_criterion_func=mifs, 
                                                                                        r=r, 
                                                                                        data=X, 
                                                                                        target_variable=y, 
                                                                                        possible_variables_index = [1,2],
                                                                                        costs=costs,
                                                                                        normalized_costs=normalized_costs,
                                                                                        prev_variables_index = [0,3])
        mifs_value = mutual_information(y, X[:,feature_index]) - mutual_information(X[:,feature_index], X[:,0])-mutual_information(X[:,feature_index], X[:,3])                                                        
        m = abs(min([
            mutual_information(y, X[:,1]) - mutual_information(X[:,1], X[:,0])-mutual_information(X[:,1], X[:,3]),
            mutual_information(y, X[:,2]) - mutual_information(X[:,2], X[:,0])-mutual_information(X[:,2], X[:,3])]))
        self.assertAlmostEqual(mifs_value, criterion_value)
        self.assertAlmostEqual((mifs_value+m)/normalized_costs[feature_index]**r, filter_value)