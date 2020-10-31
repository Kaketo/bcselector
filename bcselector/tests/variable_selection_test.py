import unittest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from bcselector.variable_selection import DiffVariableSelector, FractionVariableSelector, NoCostVariableSelector
from bcselector.data_generation import MatrixGenerator

class TestDiffVariableSelector(unittest.TestCase):
    def test_numpy_input(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        costs = [1.76,  0.19, 0.36,  0.96,  0.41,  0.17, 0.36,  0.75,  0.79, 1.38]
        lamb = 1

        dvs = DiffVariableSelector()
        dvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                lamb=lamb,
                j_criterion_func='mim')

        self.assertIsInstance(dvs.variables_selected_order, list)
        self.assertEqual(len(dvs.variables_selected_order), len(costs))
        self.assertAlmostEqual(sum(costs), sum(dvs.cost_variables_selected_order))

    def test_pandas_input(self):
        integer_matrix = pd.DataFrame(np.random.randint(0,10,(100,3)), columns=['AA','BB','CC'])
        diverse_target = pd.Series(np.random.randint(0,2,(100)))
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
    
    def test_stop_budget(self):
        integer_matrix = pd.DataFrame(np.random.randint(0,10,(100,3)), columns=['AA','BB','CC'])
        diverse_target = pd.Series(np.random.randint(0,2,(100)))
        costs = {'AA': 2, 'BB': 1.1, 'CC': 1.5}
        lamb = 1

        dvs = DiffVariableSelector()
        dvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                lamb=lamb,
                j_criterion_func='mim',
                budget=2,
                stop_budget=True)
        self.assertGreater(2, sum(dvs.cost_variables_selected_order))
        self.assertGreaterEqual(2, len(dvs.variables_selected_order))

#     def test_real_dataset(self):
#         colnames = ['Class','age','sex','steroid','antviral','fatigue','malaise','anorexia','liver_big','liver_firm',
#             'spleen_palpable','spiders','ascites','varices','bilirubin','alk_phosphate','sgot','albumin',
#             'protime','histology']
#         hepatitis = pd.read_csv('./bcselector/tests/data/hepatitis.data', header = None, names = colnames)
#         hepatitis = hepatitis.fillna(-1)
#         costs = {'age':1.00,'sex':1.00,'steroid':1.00,'antviral':1.00,'fatigue':1.00,'malaise':1.00,'anorexia':1.00,'liver_big':1.00,'liver_firm':1.00,'spleen_palpable':1.00,'spiders':1.00,'ascites':1.00,'varices':1.00,'bilirubin':7.27,'alk_phosphate':7.27,'sgot':7.27,'albumin':7.27,'protime':8.30,'histology': 1.00}
#         y_target = hepatitis['Class']
#         hepatitis.drop(['Class'], axis = 1, inplace = True)

#         # Results are calculated iteratively on paper and checked
#         # MIM
#         lamb = 0.1
#         mim_order = [0, 11, 10, 4, 18, 5, 12, 1, 9, 7, 8, 2, 3, 6, 14, 15, 16, 13, 17]
#         mim_costs = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,1., 1., 1., 7.27, 7.27, 7.27, 7.27, 8.3 ]
#         dvs = DiffVariableSelector()
#         dvs.fit(data=hepatitis,
#                 target_variable=y_target,
#                 costs=costs,
#                 lamb=lamb,
#                 j_criterion_func='mim')
#         model = LogisticRegression()
#         dvs.scoreCV(model)
#         self.assertListEqual(mim_costs, dvs.cost_variables_selected_order)
#         self.assertListEqual(mim_order, dvs.variables_selected_order)

#         # MIFS
#         lamb = 1
#         beta = 10
#         mifs_order = [0, 3, 1, 6, 2, 18, 7, 4, 9, 12, 5, 8, 11, 10, 16, 13, 17, 14, 15]
#         mifs_costs = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,7.27,7.27,8.3,7.27,7.27]
#         dvs = DiffVariableSelector()
#         dvs.fit(data=hepatitis,
#                 target_variable=y_target,
#                 costs=costs,
#                 lamb=lamb,
#                 j_criterion_func='mifs',
#                 beta=beta)
#         model = LogisticRegression()
#         dvs.scoreCV(model,seed=42)

#         self.assertListEqual(mifs_costs, dvs.cost_variables_selected_order)
#         self.assertListEqual(mifs_order, dvs.variables_selected_order)

#         # MRMR
#         lamb = 0.001
#         mrmr_order = [14, 1, 11, 4, 18, 3, 2, 10, 5, 12, 7, 9, 6, 16, 8, 13, 17, 0, 15]
#         mrmr_costs = [7.27,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,7.27,1.0,7.27,8.3,1.0,7.27]
#         dvs = DiffVariableSelector()
#         dvs.fit(data=hepatitis,
#                 target_variable=y_target,
#                 costs=costs,
#                 lamb=lamb,
#                 j_criterion_func='mrmr')
#         model = LogisticRegression()
#         dvs.scoreCV(model)
#         self.assertListEqual(mrmr_costs, dvs.cost_variables_selected_order)
#         self.assertListEqual(mrmr_order, dvs.variables_selected_order)

#         # JMI
#         lamb = 0.1
#         jmi_order = [0, 5, 10, 11, 18, 12, 4, 7, 9, 8, 2, 1, 6, 3, 15, 14, 13, 16, 17]
#         jmi_costs = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,7.27,7.27,7.27,7.27,8.3]
#         dvs = DiffVariableSelector()
#         dvs.fit(data=hepatitis,
#                 target_variable=y_target,
#                 costs=costs,
#                 lamb=lamb,
#                 j_criterion_func='jmi')
#         model = LogisticRegression()
#         dvs.scoreCV(model)
#         self.assertListEqual(jmi_costs, dvs.cost_variables_selected_order)
#         self.assertListEqual(jmi_order, dvs.variables_selected_order)

#         # CIFE
#         lamb = 0.1
#         beta = 1
#         cife_order = [0, 5, 9, 8, 10, 2, 7, 12, 6, 15, 18, 3, 11, 13, 1, 4, 16, 17, 14]
#         cife_costs = [1.,1.,1.,1.,1.,1.,1.,1.,1.,7.27,1.,1.,1.,7.27,1.,1.,7.27,8.3,7.27]
#         dvs = DiffVariableSelector()
#         dvs.fit(data=hepatitis,
#                 target_variable=y_target,
#                 costs=costs,
#                 lamb=lamb,
#                 j_criterion_func='cife',
#                 beta=beta)
#         model = LogisticRegression()
#         dvs.scoreCV(model)

#         self.assertListEqual(cife_order, dvs.variables_selected_order)
#         self.assertListEqual(cife_costs, dvs.cost_variables_selected_order)
        
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

    def test_score(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,2,(100))
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        lamb = 1

        dvs = DiffVariableSelector()
        dvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                lamb=lamb,
                j_criterion_func='mim')

        model = LogisticRegression()
        dvs.score(model, scoring_function=roc_auc_score)
        
        self.assertEqual(len(dvs.total_scores), len(costs))

    def test_run_score_before_fit(self):
        dvs = DiffVariableSelector()
        model = LogisticRegression()
        with self.assertRaises(AssertionError): dvs.score(model, scoring_function=roc_auc_score)

    def test_plot_without_comparision(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,2,(100))
        costs = [1.76,  0.19, 0.36,  0.96,  0.41,  0.17, 0.36,  0.75,  0.79, 1.38]
        lamb = 1

        dvs = DiffVariableSelector()
        dvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                lamb=lamb,
                j_criterion_func='mim')

        model = LogisticRegression()
        dvs.score(model, scoring_function=roc_auc_score)
        dvs.plot_scores(budget=1)

    def test_plot_comparision(self):
        mg = MatrixGenerator()
        X,y,costs = mg.generate(n_basic_cols=10, noise_sigmas=[0.1,1])
        lamb = 1

        dvs = DiffVariableSelector()
        dvs.fit(data=X,
                target_variable=y,
                costs=costs,
                lamb=lamb,
                j_criterion_func='mim')

        model = LogisticRegression()
        dvs.score(model, scoring_function=roc_auc_score)
        dvs.plot_scores(compare_no_cost_method=True, budget=1, model = model)

class TestFractionVariableSelector(unittest.TestCase):
    def test_numpy_input(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        r = 1

        fvs = FractionVariableSelector()
        fvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                r=r,
                j_criterion_func='mim')

        self.assertIsInstance(fvs.variables_selected_order, list)
        self.assertEqual(len(fvs.variables_selected_order), len(costs))

    def test_pandas_input(self):
        integer_matrix = pd.DataFrame(np.random.randint(0,10,(100,3)), columns=['AA','BB','CC'])
        diverse_target = np.random.randint(0,2,(100))
        costs = {'AA': 10, 'BB': 1, 'CC': 1.5}
        r = 1
        fvs = FractionVariableSelector()
        fvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                r=r,
                j_criterion_func='mim')

        self.assertIsInstance(fvs.variables_selected_order, list)
        self.assertEqual(len(fvs.variables_selected_order), len(costs))

    def test_theoretical_output(self):
        integer_matrix = np.array([[0,1,0],[0,1,0],[0,1,2],[0,1,3],[1,1,5]])
        diverse_target = np.array([0,0,0,0,1])
        costs = [1,1,1]
        r = 1

        fvs = FractionVariableSelector()
        fvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                r=r,
                j_criterion_func='mim')
        
        self.assertEqual(fvs.variables_selected_order[0], 2)

    def test_score(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,2,(100))
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]
        r = 1

        fvs = FractionVariableSelector()
        fvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                r=r,
                j_criterion_func='mim')

        model = LogisticRegression()
        fvs.score(model, scoring_function=roc_auc_score)
        
        self.assertEqual(len(fvs.total_scores), len(costs))

    def test_run_score_before_fit(self):
        fvs = FractionVariableSelector()
        model = LogisticRegression()
        with self.assertRaises(AssertionError): fvs.score(model, scoring_function=roc_auc_score)

    def test_stop_budget(self):
        integer_matrix = pd.DataFrame(np.random.randint(0,10,(100,3)), columns=['AA','BB','CC'])
        diverse_target = pd.Series(np.random.randint(0,2,(100)))
        costs = {'AA': 2, 'BB': 1.1, 'CC': 1.5}
        r = 1

        fvs = FractionVariableSelector()
        fvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                r=r,
                j_criterion_func='mim',
                budget=2,
                stop_budget=True)
        self.assertGreater(2, sum(fvs.cost_variables_selected_order))
        self.assertGreaterEqual(2, len(fvs.variables_selected_order))

class TestNoCostVariableSelector(unittest.TestCase):
    def test_numpy_input(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,10,(100))
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]

        ncvs = NoCostVariableSelector()
        ncvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                j_criterion_func='mim')

        self.assertIsInstance(ncvs.variables_selected_order, list)
        self.assertEqual(len(ncvs.variables_selected_order), len(costs))

    def test_pandas_input(self):
        integer_matrix = pd.DataFrame(np.random.randint(0,10,(100,3)), columns=['AA','BB','CC'])
        diverse_target = np.random.randint(0,2,(100))
        costs = {'AA': 10, 'BB': 1, 'CC': 1.5}

        ncvs = NoCostVariableSelector()
        ncvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                j_criterion_func='mim')

        self.assertIsInstance(ncvs.variables_selected_order, list)
        self.assertEqual(len(ncvs.variables_selected_order), len(costs))

    def test_theoretical_output(self):
        integer_matrix = np.array([[0,1,0],[0,1,0],[0,1,2],[0,1,3],[1,1,5]])
        diverse_target = np.array([0,0,0,0,1])
        costs = [1,1,1]

        ncvs = NoCostVariableSelector()
        ncvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                j_criterion_func='mim')
        
        self.assertEqual(ncvs.variables_selected_order[0], 2)

    def test_score(self):
        integer_matrix = np.random.randint(0,10,(100,10))
        diverse_target = np.random.randint(0,2,(100))
        costs = [ 1.76,  0.19, -0.36,  0.96,  0.41,  0.17, -0.36,  0.75,  0.79, -1.38]

        ncvs = NoCostVariableSelector()
        ncvs.fit(data=integer_matrix,
                target_variable=diverse_target,
                costs=costs,
                j_criterion_func='mim')

        model = LogisticRegression()
        ncvs.score(model, scoring_function=roc_auc_score)
        
        self.assertEqual(len(ncvs.total_scores), len(costs))

    def test_run_score_before_fit(self):
        ncvs = NoCostVariableSelector()
        model = LogisticRegression()
        with self.assertRaises(AssertionError): ncvs.score(model, scoring_function=roc_auc_score)

if __name__ == '__main__':
    unittest.main()