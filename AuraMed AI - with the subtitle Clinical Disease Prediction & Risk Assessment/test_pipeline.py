import unittest
import os
import shutil
import pandas as pd
from model_pipeline import DiseasePredictionPipeline

class TestDiseasePredictionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = DiseasePredictionPipeline()
        cls.test_model_dir = 'test_models'
        if not os.path.exists(cls.test_model_dir):
            os.makedirs(cls.test_model_dir)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.test_model_dir):
            shutil.rmtree(cls.test_model_dir)

    def test_heart_data_generation(self):
        df = self.pipeline.generate_heart_data(num_samples=200, random_seed=42)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 200)
        self.assertIn('Target', df.columns)
        self.assertIn('Age', df.columns)
        self.assertIn('Cholesterol', df.columns)
        self.assertIn('Thalassemia', df.columns)

    def test_diabetes_data_generation(self):
        df = self.pipeline.generate_diabetes_data(num_samples=200, random_seed=42)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 200)
        self.assertIn('Target', df.columns)
        self.assertIn('Glucose', df.columns)
        self.assertIn('BMI', df.columns)

    def test_breast_cancer_data_generation(self):
        df = self.pipeline.generate_breast_cancer_data(num_samples=200, random_seed=42)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 200)
        self.assertIn('Target', df.columns)
        self.assertIn('Radius_Mean', df.columns)
        self.assertIn('Area_Mean', df.columns)

    def test_feature_engineering(self):
        # Heart dataset
        df_heart = self.pipeline.generate_heart_data(num_samples=100, random_seed=42)
        X_heart, y_heart = self.pipeline.feature_engineering(df_heart, 'heart', is_training=True)
        self.assertIsInstance(X_heart, pd.DataFrame)
        self.assertIsInstance(y_heart, pd.Series)
        self.assertIn('BP_Cholesterol_Ratio', X_heart.columns)
        self.assertIn('Max_HR_Age_Ratio', X_heart.columns)
        self.assertIn('Sex_1', X_heart.columns)

        # Diabetes dataset
        df_diab = self.pipeline.generate_diabetes_data(num_samples=100, random_seed=42)
        X_diab, y_diab = self.pipeline.feature_engineering(df_diab, 'diabetes', is_training=True)
        self.assertIsInstance(X_diab, pd.DataFrame)
        self.assertIn('Insulin_Glucose_Ratio', X_diab.columns)

    def test_train_evaluate_and_save_load(self):
        # Heart Dataset
        df_heart = self.pipeline.generate_heart_data(num_samples=150, random_seed=42)
        results = self.pipeline.train_and_evaluate(df_heart, 'heart')
        
        # Assert metrics structure
        for model_name in ['svm', 'logistic_regression', 'random_forest', 'xgboost']:
            self.assertIn(model_name, results)
            metrics = results[model_name]
            self.assertIn('accuracy', metrics)
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
            self.assertIn('roc_auc', metrics)
            self.assertIn('confusion_matrix', metrics)
            self.assertIn('roc_curve', metrics)
            self.assertIn('feature_importances', metrics)
            
        # Test Save
        self.pipeline.save_pipeline('heart', output_dir=self.test_model_dir)
        
        # Verify file existence
        self.assertTrue(os.path.exists(os.path.join(self.test_model_dir, 'scaler_heart.joblib')))
        self.assertTrue(os.path.exists(os.path.join(self.test_model_dir, 'feature_names_heart.joblib')))
        self.assertTrue(os.path.exists(os.path.join(self.test_model_dir, 'heart_random_forest_model.joblib')))

        # Test Load
        new_pipeline = DiseasePredictionPipeline()
        new_pipeline.load_pipeline('heart', input_dir=self.test_model_dir)
        self.assertIn('heart', new_pipeline.scalers)
        self.assertIn('heart_random_forest', new_pipeline.models)

        # Test predict_single
        raw_patient = {
            'Age': 58,
            'Sex': 1,
            'Chest_Pain_Type': 2,
            'Resting_BP': 130,
            'Cholesterol': 240,
            'Fasting_Blood_Sugar': 0,
            'Rest_ECG': 1,
            'Max_Heart_Rate': 150,
            'Exercise_Angina': 0,
            'ST_Depression': 1.2,
            'ST_Slope': 1,
            'Major_Vessels': 0,
            'Thalassemia': 2
        }
        
        pred_res = new_pipeline.predict_single(raw_patient, 'heart', model_name='random_forest')
        self.assertIn('prediction', pred_res)
        self.assertIn('probability', pred_res)
        self.assertIn('decision', pred_res)
        self.assertIn(pred_res['prediction'], [0, 1])

if __name__ == '__main__':
    unittest.main()
