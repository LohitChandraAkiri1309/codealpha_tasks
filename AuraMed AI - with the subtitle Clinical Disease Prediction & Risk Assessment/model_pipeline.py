import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import joblib
import os

# Robust import of XGBoost with a Gradient Boosting fallback
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except (ImportError, ModuleNotFoundError):
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    HAS_XGB = False

class DiseasePredictionPipeline:
    def __init__(self):
        self.scalers = {}
        self.models = {}
        self.engineered_feature_names = {}

    def generate_heart_data(self, num_samples=1000, random_seed=42):
        """Generates realistic Cleveland Heart Disease dataset."""
        np.random.seed(random_seed)
        
        age = np.random.randint(29, 78, size=num_samples)
        sex = np.random.choice([0, 1], size=num_samples, p=[0.32, 0.68]) # 0=Female, 1=Male
        
        # Chest pain type: 0=Typical Angina, 1=Atypical Angina, 2=Non-Anginal, 3=Asymptomatic
        cp = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.16, 0.17, 0.28, 0.39])
        
        # Resting blood pressure (mmHg)
        trestbps = np.random.normal(131, 17, size=num_samples).astype(int)
        trestbps = np.clip(trestbps, 94, 200)
        
        # Cholesterol (mg/dl)
        chol = np.random.normal(246, 51, size=num_samples).astype(int)
        chol = np.clip(chol, 126, 564)
        
        # Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
        fbs = np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15])
        
        # Resting electrocardiographic results: 0=Normal, 1=ST-T abnormality, 2=LV hypertrophy
        restecg = np.random.choice([0, 1, 2], size=num_samples, p=[0.49, 0.50, 0.01])
        
        # Max heart rate achieved
        # Max heart rate decreases with age
        base_thalach = 220 - age + np.random.normal(0, 15, size=num_samples)
        thalach = np.clip(base_thalach, 71, 202).astype(int)
        
        # Exercise induced angina (1 = yes; 0 = no)
        exang = np.random.choice([0, 1], size=num_samples, p=[0.67, 0.33])
        
        # ST depression induced by exercise relative to rest
        oldpeak = np.random.exponential(scale=1.0, size=num_samples)
        oldpeak = np.clip(oldpeak, 0.0, 6.2).round(2)
        
        # Slope of the peak exercise ST segment: 0=Upsloping, 1=Flat, 2=Downsloping
        slope = np.random.choice([0, 1, 2], size=num_samples, p=[0.07, 0.46, 0.47])
        
        # Number of major vessels (0-4) colored by fluoroscopy
        ca = np.random.choice([0, 1, 2, 3, 4], size=num_samples, p=[0.58, 0.21, 0.13, 0.06, 0.02])
        
        # Thalassemia: 0=none, 1=normal, 2=fixed defect, 3=reversible defect
        thal = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.01, 0.06, 0.54, 0.39])
        
        # Probability equation for Heart Disease target (0=no disease, 1=disease)
        logit = (
            -2.0
            + 0.04 * (age - 50)
            + 0.8 * sex
            + np.where(cp == 3, 1.5, np.where(cp == 0, -1.0, 0.2))
            + 0.02 * (trestbps - 120)
            + 0.005 * (chol - 200)
            + 0.3 * fbs
            - 0.03 * (thalach - 150)
            + 1.2 * exang
            + 1.0 * oldpeak
            + 1.1 * ca
            + np.where(thal == 3, 1.4, np.where(thal == 1, 0.2, -0.6))
        )
        
        prob = 1.0 / (1.0 + np.exp(-logit))
        noise = np.random.normal(0, 0.2, size=num_samples)
        target = (prob + noise > 0.5).astype(int)
        
        df = pd.DataFrame({
            'Age': age,
            'Sex': sex,
            'Chest_Pain_Type': cp,
            'Resting_BP': trestbps,
            'Cholesterol': chol,
            'Fasting_Blood_Sugar': fbs,
            'Rest_ECG': restecg,
            'Max_Heart_Rate': thalach,
            'Exercise_Angina': exang,
            'ST_Depression': oldpeak,
            'ST_Slope': slope,
            'Major_Vessels': ca,
            'Thalassemia': thal,
            'Target': target
        })
        return df

    def generate_diabetes_data(self, num_samples=1000, random_seed=42):
        """Generates realistic Pima Indians Diabetes dataset."""
        np.random.seed(random_seed)
        
        pregnancies = np.random.negative_binomial(n=2, p=0.4, size=num_samples)
        pregnancies = np.clip(pregnancies, 0, 17)
        
        # Glucose concentration (mg/dl)
        glucose = np.random.normal(121, 31, size=num_samples).astype(int)
        glucose = np.clip(glucose, 44, 199)
        
        # Diastolic Blood Pressure (mmHg)
        blood_pressure = np.random.normal(69, 12, size=num_samples).astype(int)
        blood_pressure = np.clip(blood_pressure, 24, 122)
        
        # Triceps skin fold thickness (mm)
        skin_thickness = np.random.normal(20, 10, size=num_samples).astype(int)
        skin_thickness = np.clip(skin_thickness, 7, 99)
        
        # 2-Hour serum insulin (mu U/ml)
        insulin = np.random.exponential(scale=80, size=num_samples) + np.where(glucose > 130, 80, 20)
        insulin = np.clip(insulin, 14, 846).astype(int)
        
        # BMI
        bmi = np.random.normal(32, 7, size=num_samples).round(1)
        bmi = np.clip(bmi, 18.2, 67.1)
        
        # Diabetes pedigree function
        dpf = np.random.lognormal(mean=-0.8, sigma=0.4, size=num_samples).round(3)
        dpf = np.clip(dpf, 0.078, 2.42)
        
        age = np.random.randint(21, 82, size=num_samples)
        
        # Target probability logic
        logit = (
            -5.5
            + 0.12 * pregnancies
            + 0.04 * glucose
            - 0.01 * blood_pressure
            + 0.01 * skin_thickness
            + 0.001 * insulin
            + 0.09 * bmi
            + 1.5 * dpf
            + 0.03 * age
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        noise = np.random.normal(0, 0.15, size=num_samples)
        target = (prob + noise > 0.5).astype(int)
        
        df = pd.DataFrame({
            'Pregnancies': pregnancies,
            'Glucose': glucose,
            'Blood_Pressure': blood_pressure,
            'Skin_Thickness': skin_thickness,
            'Insulin': insulin,
            'BMI': bmi,
            'Diabetes_Pedigree': dpf,
            'Age': age,
            'Target': target
        })
        return df

    def generate_breast_cancer_data(self, num_samples=1000, random_seed=42):
        """Generates realistic Wisconsin Breast Cancer dataset (mean features)."""
        np.random.seed(random_seed)
        
        radius_mean = np.random.normal(14.1, 3.5, size=num_samples).round(2)
        radius_mean = np.clip(radius_mean, 6.98, 28.11)
        
        texture_mean = np.random.normal(19.3, 4.3, size=num_samples).round(2)
        texture_mean = np.clip(texture_mean, 9.71, 39.28)
        
        # Perimeter and Area are highly correlated with Radius
        perimeter_mean = (radius_mean * 6.28 + np.random.normal(0, 2, size=num_samples)).round(2)
        perimeter_mean = np.clip(perimeter_mean, 43.79, 188.5)
        
        area_mean = (np.pi * (radius_mean ** 2) + np.random.normal(0, 25, size=num_samples)).round(1)
        area_mean = np.clip(area_mean, 143.5, 2501.0)
        
        smoothness_mean = np.random.normal(0.096, 0.014, size=num_samples).round(4)
        smoothness_mean = np.clip(smoothness_mean, 0.053, 0.163)
        
        compactness_mean = np.random.normal(0.104, 0.052, size=num_samples).round(4)
        compactness_mean = np.clip(compactness_mean, 0.019, 0.345)
        
        concavity_mean = np.random.normal(0.088, 0.079, size=num_samples).round(4)
        concavity_mean = np.clip(concavity_mean, 0.0, 0.427)
        
        concave_points_mean = np.random.normal(0.048, 0.038, size=num_samples).round(4)
        concave_points_mean = np.clip(concave_points_mean, 0.0, 0.201)
        
        symmetry_mean = np.random.normal(0.181, 0.027, size=num_samples).round(4)
        symmetry_mean = np.clip(symmetry_mean, 0.106, 0.304)
        
        fractal_dimension_mean = np.random.normal(0.062, 0.007, size=num_samples).round(5)
        fractal_dimension_mean = np.clip(fractal_dimension_mean, 0.05, 0.097)
        
        # Malignancy logit equation
        logit = (
            -15.0
            + 0.5 * radius_mean
            + 0.1 * texture_mean
            + 0.05 * perimeter_mean
            + 0.002 * area_mean
            + 25 * smoothness_mean
            + 15 * compactness_mean
            + 20 * concavity_mean
            + 40 * concave_points_mean
            + 5 * symmetry_mean
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        noise = np.random.normal(0, 0.1, size=num_samples)
        target = (prob + noise > 0.5).astype(int)
        
        df = pd.DataFrame({
            'Radius_Mean': radius_mean,
            'Texture_Mean': texture_mean,
            'Perimeter_Mean': perimeter_mean,
            'Area_Mean': area_mean,
            'Smoothness_Mean': smoothness_mean,
            'Compactness_Mean': compactness_mean,
            'Concavity_Mean': concavity_mean,
            'Concave_Points_Mean': concave_points_mean,
            'Symmetry_Mean': symmetry_mean,
            'Fractal_Dimension_Mean': fractal_dimension_mean,
            'Target': target
        })
        return df

    def feature_engineering(self, df, dataset_name, is_training=True):
        """Preprocesses dataset, runs feature scaling, and feature engineering."""
        df_eng = df.copy()
        
        if dataset_name == 'heart':
            df_eng['BP_Cholesterol_Ratio'] = df_eng['Resting_BP'] / (df_eng['Cholesterol'] + 1e-5)
            df_eng['Max_HR_Age_Ratio'] = df_eng['Max_Heart_Rate'] / (df_eng['Age'] + 1e-5)
            
            numeric_cols = [
                'Age', 'Resting_BP', 'Cholesterol', 'Max_Heart_Rate', 
                'ST_Depression', 'Major_Vessels', 'BP_Cholesterol_Ratio', 'Max_HR_Age_Ratio'
            ]
            
            categorical_configs = {
                'Sex': [0, 1],
                'Chest_Pain_Type': [0, 1, 2, 3],
                'Fasting_Blood_Sugar': [0, 1],
                'Rest_ECG': [0, 1, 2],
                'Exercise_Angina': [0, 1],
                'ST_Slope': [0, 1, 2],
                'Thalassemia': [0, 1, 2, 3]
            }
            
            cat_cols = []
            for col, levels in categorical_configs.items():
                for level in levels:
                    col_name = f"{col}_{level}"
                    df_eng[col_name] = (df_eng[col] == level).astype(float)
                    cat_cols.append(col_name)
                    
            self.engineered_feature_names[dataset_name] = numeric_cols + cat_cols
            
        elif dataset_name == 'diabetes':
            df_eng['Insulin_Glucose_Ratio'] = df_eng['Insulin'] / (df_eng['Glucose'] + 1.0)
            df_eng['BMI_Age_Ratio'] = df_eng['BMI'] / (df_eng['Age'] + 1e-5)
            
            numeric_cols = [
                'Pregnancies', 'Glucose', 'Blood_Pressure', 'Skin_Thickness', 
                'Insulin', 'BMI', 'Diabetes_Pedigree', 'Age', 
                'Insulin_Glucose_Ratio', 'BMI_Age_Ratio'
            ]
            cat_cols = []
            self.engineered_feature_names[dataset_name] = numeric_cols
            
        elif dataset_name == 'breast_cancer':
            df_eng['Area_Perimeter_Ratio'] = df_eng['Area_Mean'] / (df_eng['Perimeter_Mean'] + 1e-5)
            df_eng['Compactness_Concavity_Product'] = df_eng['Compactness_Mean'] * df_eng['Concavity_Mean']
            
            numeric_cols = [
                'Radius_Mean', 'Texture_Mean', 'Perimeter_Mean', 'Area_Mean', 
                'Smoothness_Mean', 'Compactness_Mean', 'Concavity_Mean', 
                'Concave_Points_Mean', 'Symmetry_Mean', 'Fractal_Dimension_Mean',
                'Area_Perimeter_Ratio', 'Compactness_Concavity_Product'
            ]
            cat_cols = []
            self.engineered_feature_names[dataset_name] = numeric_cols
            
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")
            
        features = self.engineered_feature_names[dataset_name]
        
        # Separate X and y
        if 'Target' in df_eng.columns:
            X = df_eng[features]
            y = df_eng['Target']
        else:
            X = df_eng[features]
            y = None
            
        # Scale numeric features
        if is_training:
            scaler = StandardScaler()
            scaler.fit(X[numeric_cols])
            self.scalers[dataset_name] = scaler
            
        scaler = self.scalers[dataset_name]
        X_numeric_scaled = scaler.transform(X[numeric_cols])
        
        if len(cat_cols) > 0:
            X_scaled = np.hstack((X_numeric_scaled, X[cat_cols].values))
        else:
            X_scaled = X_numeric_scaled
            
        X_df = pd.DataFrame(X_scaled, columns=features)
        
        return X_df, y

    def train_and_evaluate(self, df, dataset_name, svm_params=None, lr_params=None, rf_params=None, xgb_params=None):
        """Trains SVM, Logistic Regression, Random Forest, and XGBoost/GBDT and evaluates performance."""
        X, y = self.feature_engineering(df, dataset_name, is_training=True)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        svm_args = svm_params or {'C': 1.0, 'kernel': 'rbf', 'probability': True, 'random_state': 42}
        lr_args = lr_params or {'C': 1.0, 'max_iter': 1000, 'random_state': 42}
        rf_args = rf_params or {'n_estimators': 100, 'max_depth': 6, 'random_state': 42}
        
        if HAS_XGB:
            xgb_args = xgb_params or {'n_estimators': 80, 'max_depth': 4, 'learning_rate': 0.1, 'random_state': 42, 'eval_metric': 'logloss'}
        else:
            xgb_args = xgb_params or {'n_estimators': 80, 'max_depth': 4, 'learning_rate': 0.1, 'random_state': 42}
            
        self.models[f'{dataset_name}_svm'] = SVC(**svm_args)
        self.models[f'{dataset_name}_logistic_regression'] = LogisticRegression(**lr_args)
        self.models[f'{dataset_name}_random_forest'] = RandomForestClassifier(**rf_args)
        self.models[f'{dataset_name}_xgboost'] = XGBClassifier(**xgb_args)
        
        results = {}
        model_keys = ['svm', 'logistic_regression', 'random_forest', 'xgboost']
        
        for key in model_keys:
            full_key = f"{dataset_name}_{key}"
            model = self.models[full_key]
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_prob)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            cm_dict = {
                'tn': int(cm[0, 0]),
                'fp': int(cm[0, 1]),
                'fn': int(cm[1, 0]),
                'tp': int(cm[1, 1])
            }
            
            # ROC Curve
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            step = max(1, len(fpr) // 40)
            roc_points = [{'fpr': float(f), 'tpr': float(t)} for f, t in zip(fpr[::step], tpr[::step])]
            if len(roc_points) == 0 or roc_points[-1] != {'fpr': 1.0, 'tpr': 1.0}:
                roc_points.append({'fpr': 1.0, 'tpr': 1.0})
                
            # Feature importance
            features = self.engineered_feature_names[dataset_name]
            importances = {}
            if key == 'logistic_regression':
                coefs = model.coef_[0]
                for f_name, coef in zip(features, coefs):
                    importances[f_name] = float(coef)
            elif key == 'svm':
                if svm_args.get('kernel', 'rbf') == 'linear':
                    coefs = model.coef_[0]
                    for f_name, coef in zip(features, coefs):
                        importances[f_name] = float(coef)
                else:
                    for f_name in features:
                        importances[f_name] = 0.0
            elif key in ['random_forest', 'xgboost']:
                imp = model.feature_importances_
                for f_name, score in zip(features, imp):
                    importances[f_name] = float(score)
                    
            results[key] = {
                'accuracy': float(acc),
                'precision': float(prec),
                'recall': float(rec),
                'f1_score': float(f1),
                'roc_auc': float(auc),
                'confusion_matrix': cm_dict,
                'roc_curve': roc_points,
                'feature_importances': importances
            }
            
        return results

    def save_pipeline(self, dataset_name, output_dir='models'):
        """Saves scaler, model feature names, and 4 models for the given dataset."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        joblib.dump(self.scalers[dataset_name], os.path.join(output_dir, f'scaler_{dataset_name}.joblib'))
        joblib.dump(self.engineered_feature_names[dataset_name], os.path.join(output_dir, f'feature_names_{dataset_name}.joblib'))
        
        for key in ['svm', 'logistic_regression', 'random_forest', 'xgboost']:
            model_key = f"{dataset_name}_{key}"
            if model_key in self.models:
                joblib.dump(self.models[model_key], os.path.join(output_dir, f'{model_key}_model.joblib'))

    def load_pipeline(self, dataset_name, input_dir='models'):
        """Loads scaler, feature names, and models for the given dataset."""
        self.scalers[dataset_name] = joblib.load(os.path.join(input_dir, f'scaler_{dataset_name}.joblib'))
        self.engineered_feature_names[dataset_name] = joblib.load(os.path.join(input_dir, f'feature_names_{dataset_name}.joblib'))
        
        for key in ['svm', 'logistic_regression', 'random_forest', 'xgboost']:
            model_key = f"{dataset_name}_{key}"
            path = os.path.join(input_dir, f'{model_key}_model.joblib')
            if os.path.exists(path):
                self.models[model_key] = joblib.load(path)

    def predict_single(self, raw_input, dataset_name, model_name='random_forest'):
        """Runs predictions for a single patient dictionary."""
        df_single = pd.DataFrame([raw_input])
        X_single, _ = self.feature_engineering(df_single, dataset_name, is_training=False)
        
        full_model_key = f"{dataset_name}_{model_name}"
        if full_model_key not in self.models:
            raise ValueError(f"Model {full_model_key} is not loaded or trained.")
            
        model = self.models[full_model_key]
        
        prediction = int(model.predict(X_single)[0])
        probability = float(model.predict_proba(X_single)[0, 1])
        
        return {
            'prediction': prediction,
            'probability': probability,
            'decision': 'High Risk / Positive' if prediction == 1 else 'Low Risk / Negative'
        }

if __name__ == '__main__':
    pipeline = DiseasePredictionPipeline()
    print("Testing pipeline on Heart Dataset...")
    df_heart = pipeline.generate_heart_data(num_samples=1000)
    res_heart = pipeline.train_and_evaluate(df_heart, 'heart')
    print("Heart RF Accuracy:", res_heart['random_forest']['accuracy'])
    pipeline.save_pipeline('heart')
    
    print("\nTesting pipeline on Diabetes Dataset...")
    df_diab = pipeline.generate_diabetes_data(num_samples=1000)
    res_diab = pipeline.train_and_evaluate(df_diab, 'diabetes')
    print("Diabetes RF Accuracy:", res_diab['random_forest']['accuracy'])
    pipeline.save_pipeline('diabetes')
    
    print("\nTesting pipeline on Breast Cancer Dataset...")
    df_cancer = pipeline.generate_breast_cancer_data(num_samples=1000)
    res_cancer = pipeline.train_and_evaluate(df_cancer, 'breast_cancer')
    print("Breast Cancer RF Accuracy:", res_cancer['random_forest']['accuracy'])
    pipeline.save_pipeline('breast_cancer')
    print("\nAll pipelines tested and saved successfully!")
