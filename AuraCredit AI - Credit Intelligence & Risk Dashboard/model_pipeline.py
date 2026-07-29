import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import joblib
import os

class CreditScoringPipeline:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_names = None
        self.engineered_feature_names = None

    def generate_synthetic_data(self, num_samples=1000, random_seed=42):
        """Generates a realistic synthetic credit risk dataset."""
        np.random.seed(random_seed)
        
        # 1. Base features
        age = np.random.randint(18, 70, size=num_samples)
        
        # Income depends slightly on age
        base_income = np.random.lognormal(mean=10.8, sigma=0.5, size=num_samples)
        age_modifier = 1.0 + (age - 18) / 52.0 * 0.5  # older applicants tend to earn slightly more
        annual_income = base_income * age_modifier
        annual_income = np.clip(annual_income, 15000, 250000)
        
        # Employment duration (max age - 18)
        max_emp = np.clip(age - 18, 0, 45)
        employment_duration = np.array([np.random.randint(0, int(max_val) + 1) for max_val in max_emp])
        
        # Home ownership: Rent (40%), Mortgage (45%), Own (15%)
        home_ownership = np.random.choice(['Rent', 'Mortgage', 'Own'], size=num_samples, p=[0.4, 0.45, 0.15])
        
        # Credit limit depends heavily on income and age
        base_limit = annual_income * np.random.uniform(0.1, 0.4, size=num_samples)
        credit_limit = np.clip(base_limit, 1000, 50000)
        
        # Current balance as fraction of credit limit (utilization)
        utilization_rate = np.random.beta(a=2, b=5, size=num_samples)
        current_balance = credit_limit * utilization_rate
        
        # Monthly debt payments (DTI calculation)
        base_debt = (annual_income / 12) * np.random.uniform(0.05, 0.5, size=num_samples)
        monthly_debt = np.clip(base_debt, 50, 5000)
        
        # Delinquency history (0 to 10 delinquencies)
        # Higher delinquencies for high utilization, lower income
        delinquency_prob = np.clip(utilization_rate * 0.4 + (20000 / annual_income) * 0.2, 0.01, 0.99)
        delinquencies = np.random.binomial(n=10, p=delinquency_prob, size=num_samples)
        
        # 2. Compute Target (Creditworthiness) based on logistic equation + noise
        dti = (monthly_debt * 12) / annual_income
        util = current_balance / credit_limit
        
        logit = (
            2.0 
            + 1.5 * (annual_income / 100000.0) 
            + 0.8 * (employment_duration / 10.0)
            + np.where(home_ownership == 'Own', 1.2, np.where(home_ownership == 'Mortgage', 0.4, -0.6))
            - 5.5 * util 
            - 5.0 * dti 
            - 1.8 * delinquencies 
            - 0.5 * (age < 25).astype(float)
        )
        
        # Probability of creditworthiness
        prob = 1.0 / (1.0 + np.exp(-logit))
        
        # Binary target (1 = Creditworthy, 0 = High Risk)
        noise = np.random.normal(0, 0.15, size=num_samples)
        target = (prob + noise > 0.5).astype(int)
        
        # Construct DataFrame
        df = pd.DataFrame({
            'Age': age,
            'Annual_Income': np.round(annual_income, 2),
            'Credit_Limit': np.round(credit_limit, 2),
            'Current_Balance': np.round(current_balance, 2),
            'Monthly_Debt_Payments': np.round(monthly_debt, 2),
            'Employment_Duration_Years': employment_duration,
            'Payment_History_Delinquencies': delinquencies,
            'Home_Ownership': home_ownership,
            'Creditworthy': target
        })
        
        return df

    def feature_engineering(self, df, is_training=True):
        """Creates engineered features and pre-processes dataset."""
        df_eng = df.copy()
        
        # 1. Ratios
        df_eng['Credit_Utilization_Ratio'] = df_eng['Current_Balance'] / df_eng['Credit_Limit']
        # Handle division by zero if credit limit is 0
        df_eng['Credit_Utilization_Ratio'] = df_eng['Credit_Utilization_Ratio'].fillna(0)
        
        df_eng['Debt_to_Income_Ratio'] = (df_eng['Monthly_Debt_Payments'] * 12) / df_eng['Annual_Income']
        df_eng['Payment_to_Income_Ratio'] = df_eng['Monthly_Debt_Payments'] / (df_eng['Annual_Income'] / 12)
        
        # 2. Categorical Encoding (Home_Ownership)
        # Ensure consistent columns regardless of sample properties
        for cat in ['Own', 'Mortgage', 'Rent']:
            df_eng[f'Home_Ownership_{cat}'] = (df_eng['Home_Ownership'] == cat).astype(int)
            
        # Drop original Home_Ownership
        df_eng = df_eng.drop(columns=['Home_Ownership'])
        
        # Select features to scale
        numeric_cols = [
            'Age', 'Annual_Income', 'Credit_Limit', 'Current_Balance', 
            'Monthly_Debt_Payments', 'Employment_Duration_Years', 
            'Payment_History_Delinquencies', 'Credit_Utilization_Ratio', 
            'Debt_to_Income_Ratio', 'Payment_to_Income_Ratio'
        ]
        
        categorical_cols = ['Home_Ownership_Own', 'Home_Ownership_Mortgage', 'Home_Ownership_Rent']
        
        self.engineered_feature_names = numeric_cols + categorical_cols
        
        # Separate features (X) and target (y)
        if 'Creditworthy' in df_eng.columns:
            X = df_eng[self.engineered_feature_names]
            y = df_eng['Creditworthy']
        else:
            X = df_eng[self.engineered_feature_names]
            y = None
            
        # Scale numeric features
        if is_training:
            X_numeric_scaled = self.scaler.fit_transform(X[numeric_cols])
        else:
            X_numeric_scaled = self.scaler.transform(X[numeric_cols])
            
        X_scaled = np.hstack((X_numeric_scaled, X[categorical_cols].values))
        
        # Return features as a clean DataFrame/numpy array
        X_df = pd.DataFrame(X_scaled, columns=self.engineered_feature_names)
        
        return X_df, y

    def train_and_evaluate(self, df, lr_params=None, dt_params=None, rf_params=None):
        """Trains Logistic Regression, Decision Tree, and Random Forest on data and evaluates performance."""
        # Feature engineering
        X, y = self.feature_engineering(df, is_training=True)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Define model configurations
        lr_args = lr_params or {'C': 1.0, 'max_iter': 1000, 'random_state': 42}
        dt_args = dt_params or {'max_depth': 5, 'min_samples_split': 10, 'random_state': 42}
        rf_args = rf_params or {'n_estimators': 100, 'max_depth': 6, 'random_state': 42}
        
        self.models['logistic_regression'] = LogisticRegression(**lr_args)
        self.models['decision_tree'] = DecisionTreeClassifier(**dt_args)
        self.models['random_forest'] = RandomForestClassifier(**rf_args)
        
        results = {}
        
        for name, model in self.models.items():
            # Fit model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
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
            # Sample coordinates to avoid sending too much data to front-end
            step = max(1, len(fpr) // 50)
            roc_points = [{'fpr': float(f), 'tpr': float(t)} for f, t in zip(fpr[::step], tpr[::step])]
            # Ensure final point (1.0, 1.0) is included
            if len(roc_points) == 0 or roc_points[-1] != {'fpr': 1.0, 'tpr': 1.0}:
                roc_points.append({'fpr': 1.0, 'tpr': 1.0})
                
            # Feature importance / Coefficients
            importances = {}
            if name == 'logistic_regression':
                coefs = model.coef_[0]
                for f_name, coef in zip(self.engineered_feature_names, coefs):
                    importances[f_name] = float(coef)
            else:
                imp = model.feature_importances_
                for f_name, score in zip(self.engineered_feature_names, imp):
                    importances[f_name] = float(score)
            
            results[name] = {
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

    def save_pipeline(self, output_dir='models'):
        """Saves current trained models and preprocessors to output directory."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.joblib'))
        joblib.dump(self.engineered_feature_names, os.path.join(output_dir, 'feature_names.joblib'))
        
        for name, model in self.models.items():
            joblib.dump(model, os.path.join(output_dir, f'{name}_model.joblib'))

    def load_pipeline(self, input_dir='models'):
        """Loads models and preprocessor from input directory."""
        self.scaler = joblib.load(os.path.join(input_dir, 'scaler.joblib'))
        self.engineered_feature_names = joblib.load(os.path.join(input_dir, 'feature_names.joblib'))
        
        for name in ['logistic_regression', 'decision_tree', 'random_forest']:
            path = os.path.join(input_dir, f'{name}_model.joblib')
            if os.path.exists(path):
                self.models[name] = joblib.load(path)

    def predict_single(self, raw_input, model_name='random_forest'):
        """Runs predictions for a single applicant dictionary."""
        # Convert dictionary to DataFrame
        df_single = pd.DataFrame([raw_input])
        
        # Run same feature engineering
        X_single, _ = self.feature_engineering(df_single, is_training=False)
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} is not loaded or trained.")
            
        model = self.models[model_name]
        
        prediction = int(model.predict(X_single)[0])
        probability = float(model.predict_proba(X_single)[0, 1])
        
        return {
            'prediction': prediction,
            'probability': probability,
            'decision': 'Creditworthy' if prediction == 1 else 'High Risk'
        }

if __name__ == '__main__':
    pipeline = CreditScoringPipeline()
    df = pipeline.generate_synthetic_data(num_samples=1000)
    print("Generated data shape:", df.shape)
    print("Class breakdown:\n", df['Creditworthy'].value_counts(normalize=True))
    
    results = pipeline.train_and_evaluate(df)
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()} Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
    
    pipeline.save_pipeline()
    print("\nPipeline saved successfully.")
