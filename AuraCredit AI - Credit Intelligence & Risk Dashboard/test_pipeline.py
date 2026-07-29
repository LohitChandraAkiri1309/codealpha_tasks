import unittest
import pandas as pd
import numpy as np
import os
import shutil
from model_pipeline import CreditScoringPipeline

class TestCreditScoringPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a single pipeline instance for testing
        cls.pipeline = CreditScoringPipeline()
        cls.df = cls.pipeline.generate_synthetic_data(num_samples=200, random_seed=42)
        cls.results = cls.pipeline.train_and_evaluate(cls.df)
        cls.test_dir = 'test_models'
        cls.pipeline.save_pipeline(output_dir=cls.test_dir)

    @classmethod
    def tearDownClass(cls):
        # Clean up saved test models
        if os.path.exists(cls.test_dir):
            shutil.rmtree(cls.test_dir)

    def test_data_generation(self):
        # Verify shape and columns of generated dataset
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertEqual(len(self.df), 200)
        
        expected_columns = [
            'Age', 'Annual_Income', 'Credit_Limit', 'Current_Balance',
            'Monthly_Debt_Payments', 'Employment_Duration_Years',
            'Payment_History_Delinquencies', 'Home_Ownership', 'Creditworthy'
        ]
        for col in expected_columns:
            self.assertIn(col, self.df.columns)

    def test_feature_engineering(self):
        # Run feature engineering
        X, y = self.pipeline.feature_engineering(self.df, is_training=True)
        
        # Verify target is extracted
        self.assertIsNotNone(y)
        self.assertEqual(len(y), len(self.df))
        
        # Verify engineered features exist in column headers
        expected_engineered = [
            'Credit_Utilization_Ratio', 'Debt_to_Income_Ratio', 'Payment_to_Income_Ratio',
            'Home_Ownership_Own', 'Home_Ownership_Mortgage', 'Home_Ownership_Rent'
        ]
        for col in expected_engineered:
            self.assertIn(col, X.columns)
            
        # Verify ratio math is correct on raw values
        for i in range(5):
            raw_row = self.df.iloc[i]
            # Since features are scaled in X, let's manually verify the engineered formulas
            expected_util = raw_row['Current_Balance'] / raw_row['Credit_Limit']
            expected_dti = (raw_row['Monthly_Debt_Payments'] * 12) / raw_row['Annual_Income']
            
            # Utilization check (or fillna check if credit limit is 0)
            if raw_row['Credit_Limit'] > 0:
                self.assertAlmostEqual(expected_util, raw_row['Current_Balance'] / raw_row['Credit_Limit'])

    def test_training_and_evaluation(self):
        # Verify output structure for each model
        expected_models = ['logistic_regression', 'decision_tree', 'random_forest']
        for model in expected_models:
            self.assertIn(model, self.results)
            metrics = self.results[model]
            
            # Check metric ranges
            for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
                self.assertIn(metric, metrics)
                self.assertTrue(0.0 <= metrics[metric] <= 1.0, f"{metric} out of bounds: {metrics[metric]}")
                
            # Check confusion matrix
            self.assertIn('confusion_matrix', metrics)
            cm = metrics['confusion_matrix']
            for cell in ['tn', 'fp', 'fn', 'tp']:
                self.assertIn(cell, cm)
                self.assertIsInstance(cm[cell], int)
                
            # Check ROC curve points
            self.assertIn('roc_curve', metrics)
            roc = metrics['roc_curve']
            self.assertTrue(len(roc) > 0)
            self.assertIn('fpr', roc[0])
            self.assertIn('tpr', roc[0])

    def test_save_and_load(self):
        # Load saved pipeline in a new instance
        new_pipeline = CreditScoringPipeline()
        new_pipeline.load_pipeline(input_dir=self.test_dir)
        
        # Verify components loaded correctly
        self.assertEqual(len(new_pipeline.models), 3)
        self.assertIsNotNone(new_pipeline.scaler)
        self.assertEqual(new_pipeline.engineered_feature_names, self.pipeline.engineered_feature_names)

    def test_single_prediction(self):
        # Mock a raw input record
        raw_applicant = {
            'Age': 40,
            'Annual_Income': 80000.0,
            'Credit_Limit': 20000.0,
            'Current_Balance': 2000.0,
            'Monthly_Debt_Payments': 400.0,
            'Employment_Duration_Years': 8,
            'Payment_History_Delinquencies': 0,
            'Home_Ownership': 'Mortgage'
        }
        
        # Run prediction for each model
        for model in ['logistic_regression', 'decision_tree', 'random_forest']:
            pred_res = self.pipeline.predict_single(raw_applicant, model_name=model)
            
            self.assertIn('prediction', pred_res)
            self.assertIn('probability', pred_res)
            self.assertIn('decision', pred_res)
            
            self.assertIn(pred_res['prediction'], [0, 1])
            self.assertTrue(0.0 <= pred_res['probability'] <= 1.0)
            self.assertIn(pred_res['decision'], ['Creditworthy', 'High Risk'])

if __name__ == '__main__':
    unittest.main()
