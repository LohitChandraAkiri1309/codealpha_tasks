from flask import Flask, request, jsonify, render_template
import os
import pandas as pd
from model_pipeline import DiseasePredictionPipeline

app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialize pipeline
pipeline = DiseasePredictionPipeline()

# Generate and cache synthetic datasets on startup
print("Generating synthetic datasets...")
datasets = {
    'heart': pipeline.generate_heart_data(num_samples=1000),
    'diabetes': pipeline.generate_diabetes_data(num_samples=1000),
    'breast_cancer': pipeline.generate_breast_cancer_data(num_samples=1000)
}

# Pre-train models for all datasets on startup
print("Pre-training models on startup...")
training_results = {
    'heart': pipeline.train_and_evaluate(datasets['heart'], 'heart'),
    'diabetes': pipeline.train_and_evaluate(datasets['diabetes'], 'diabetes'),
    'breast_cancer': pipeline.train_and_evaluate(datasets['breast_cancer'], 'breast_cancer')
}
for name in datasets.keys():
    pipeline.save_pipeline(name)
print("All startup models trained and saved successfully.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    try:
        dataset_name = request.args.get('dataset', 'heart')
        if dataset_name not in datasets:
            return jsonify({'success': False, 'error': f"Unknown dataset: {dataset_name}"}), 400
            
        df = datasets[dataset_name]
        sample_df = df.head(50).copy()
        
        # Add engineered features to the preview for transparency
        if dataset_name == 'heart':
            sample_df['BP_Cholesterol_Ratio'] = (sample_df['Resting_BP'] / sample_df['Cholesterol']).round(4)
            sample_df['Max_HR_Age_Ratio'] = (sample_df['Max_Heart_Rate'] / sample_df['Age']).round(4)
            
            stats = {
                'total_patients': len(df),
                'avg_age': float(df['Age'].mean()),
                'avg_resting_bp': float(df['Resting_BP'].mean()),
                'avg_cholesterol': float(df['Cholesterol'].mean()),
                'positive_pct': float((df['Target'].sum() / len(df)) * 100)
            }
        elif dataset_name == 'diabetes':
            sample_df['Insulin_Glucose_Ratio'] = (sample_df['Insulin'] / (sample_df['Glucose'] + 1)).round(4)
            sample_df['BMI_Age_Ratio'] = (sample_df['BMI'] / sample_df['Age']).round(4)
            
            stats = {
                'total_patients': len(df),
                'avg_glucose': float(df['Glucose'].mean()),
                'avg_bmi': float(df['BMI'].mean()),
                'avg_age': float(df['Age'].mean()),
                'positive_pct': float((df['Target'].sum() / len(df)) * 100)
            }
        else: # breast_cancer
            sample_df['Area_Perimeter_Ratio'] = (sample_df['Area_Mean'] / sample_df['Perimeter_Mean']).round(4)
            sample_df['Compactness_Concavity_Product'] = (sample_df['Compactness_Mean'] * sample_df['Concavity_Mean']).round(4)
            
            stats = {
                'total_patients': len(df),
                'avg_radius': float(df['Radius_Mean'].mean()),
                'avg_area': float(df['Area_Mean'].mean()),
                'avg_concavity': float(df['Concavity_Mean'].mean()),
                'positive_pct': float((df['Target'].sum() / len(df)) * 100)
            }
            
        records = sample_df.to_dict(orient='records')
        return jsonify({
            'success': True,
            'data': records,
            'summary_stats': stats,
            'default_results': training_results[dataset_name]
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_models():
    try:
        params = request.get_json() or {}
        dataset_name = params.get('dataset', 'heart')
        
        if dataset_name not in datasets:
            return jsonify({'success': False, 'error': f"Unknown dataset: {dataset_name}"}), 400
            
        # Parse parameters for all four models
        svm_params = params.get('svm', {})
        lr_params = params.get('logistic_regression', {})
        rf_params = params.get('random_forest', {})
        xgb_params = params.get('xgboost', {})
        
        # Clean & structure args
        svm_args = {
            'C': float(svm_params.get('C', 1.0)),
            'kernel': str(svm_params.get('kernel', 'rbf')),
            'probability': True,
            'random_state': 42
        }
        
        lr_args = {
            'C': float(lr_params.get('C', 1.0)),
            'max_iter': int(lr_params.get('max_iter', 1000)),
            'random_state': 42
        }
        
        rf_args = {
            'n_estimators': int(rf_params.get('n_estimators', 100)),
            'max_depth': int(rf_params.get('max_depth', 6)) if rf_params.get('max_depth') else None,
            'random_state': 42
        }
        
        xgb_args = {
            'n_estimators': int(xgb_params.get('n_estimators', 80)),
            'max_depth': int(xgb_params.get('max_depth', 4)) if xgb_params.get('max_depth') else None,
            'learning_rate': float(xgb_params.get('learning_rate', 0.1)),
            'random_state': 42
        }
        
        # Train & evaluate on selected dataset
        results = pipeline.train_and_evaluate(
            datasets[dataset_name],
            dataset_name=dataset_name,
            svm_params=svm_args,
            lr_params=lr_args,
            rf_params=rf_args,
            xgb_params=xgb_args
        )
        
        # Save pipeline models
        pipeline.save_pipeline(dataset_name)
        
        # Cache results
        training_results[dataset_name] = results
        
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        if not input_data:
            return jsonify({'success': False, 'error': 'No input data provided'}), 400
            
        dataset_name = input_data.get('dataset', 'heart')
        model_name = input_data.get('model_name', 'random_forest')
        
        if dataset_name not in datasets:
            return jsonify({'success': False, 'error': f"Unknown dataset: {dataset_name}"}), 400
            
        raw_input = {}
        explanations = []
        
        if dataset_name == 'heart':
            raw_input = {
                'Age': int(input_data['age']),
                'Sex': int(input_data['sex']),
                'Chest_Pain_Type': int(input_data['cp']),
                'Resting_BP': float(input_data['trestbps']),
                'Cholesterol': float(input_data['chol']),
                'Fasting_Blood_Sugar': int(input_data['fbs']),
                'Rest_ECG': int(input_data['restecg']),
                'Max_Heart_Rate': float(input_data['thalach']),
                'Exercise_Angina': int(input_data['exang']),
                'ST_Depression': float(input_data['oldpeak']),
                'ST_Slope': int(input_data['slope']),
                'Major_Vessels': int(input_data['ca']),
                'Thalassemia': int(input_data['thal'])
            }
            
            # Clinical explanations based on input thresholds
            if raw_input['Age'] > 55:
                explanations.append("Age above 55 increases general cardiovascular vulnerability.")
            if raw_input['Sex'] == 1:
                explanations.append("Male biological sex is statistically correlated with a higher incidence of coronary events in the reference Cleveland cohort.")
            if raw_input['Chest_Pain_Type'] == 3:
                explanations.append("Asymptomatic chest pain (cp=3) is strongly associated with silent ischemic heart disease.")
            elif raw_input['Chest_Pain_Type'] == 0:
                explanations.append("Typical angina symptoms points to exercise-induced coronary artery narrowing.")
            if raw_input['ST_Depression'] > 1.5:
                explanations.append(f"ST segment depression of {raw_input['ST_Depression']} mm is a critical indicator of myocardial ischemia during physical stress.")
            if raw_input['Resting_BP'] > 140:
                explanations.append(f"Resting blood pressure of {int(raw_input['Resting_BP'])} mmHg indicates clinical hypertension, stressing heart muscle.")
            if raw_input['Cholesterol'] > 240:
                explanations.append(f"Elevated cholesterol ({int(raw_input['Cholesterol'])} mg/dl) increases atherosclerotic plaque formation risk.")
            if raw_input['Thalassemia'] == 3:
                explanations.append("Reversible thalassemia perfusion defect indicates transient myocardial blood flow restriction.")
            if raw_input['Major_Vessels'] > 0:
                explanations.append(f"Fluoroscopy detected {raw_input['Major_Vessels']} occluded vessel(s), indicating advanced atherosclerosis.")
                
        elif dataset_name == 'diabetes':
            raw_input = {
                'Pregnancies': int(input_data['pregnancies']),
                'Glucose': float(input_data['glucose']),
                'Blood_Pressure': float(input_data['blood_pressure']),
                'Skin_Thickness': float(input_data['skin_thickness']),
                'Insulin': float(input_data['insulin']),
                'BMI': float(input_data['bmi']),
                'Diabetes_Pedigree': float(input_data['dpf']),
                'Age': int(input_data['age'])
            }
            
            # Clinical explanations
            if raw_input['Glucose'] > 125:
                explanations.append(f"Elevated glucose ({int(raw_input['Glucose'])} mg/dl) indicates impaired glucose tolerance or potential diabetes.")
            elif raw_input['Glucose'] < 90:
                explanations.append(f"Optimal fasting glucose ({int(raw_input['Glucose'])} mg/dl) is within healthy bounds.")
                
            if raw_input['BMI'] > 30:
                explanations.append(f"Obese BMI range ({raw_input['BMI']}) is strongly associated with elevated peripheral insulin resistance.")
            elif raw_input['BMI'] < 25:
                explanations.append(f"Healthy BMI range ({raw_input['BMI']}) reduces risk of metabolic syndrome.")
                
            if raw_input['Pregnancies'] > 4:
                explanations.append(f"High pregnancy count ({raw_input['Pregnancies']}) is correlated with an increased risk of long-term beta-cell dysfunction.")
            if raw_input['Age'] > 45:
                explanations.append("Age over 45 is a primary risk factor for type 2 diabetes mellitus due to metabolic slowing.")
            if raw_input['Diabetes_Pedigree'] > 0.5:
                explanations.append(f"Elevated diabetes pedigree score ({raw_input['Diabetes_Pedigree']}) highlights hereditary risk factors.")
                
        elif dataset_name == 'breast_cancer':
            raw_input = {
                'Radius_Mean': float(input_data['radius_mean']),
                'Texture_Mean': float(input_data['texture_mean']),
                'Perimeter_Mean': float(input_data['perimeter_mean']),
                'Area_Mean': float(input_data['area_mean']),
                'Smoothness_Mean': float(input_data['smoothness_mean']),
                'Compactness_Mean': float(input_data['compactness_mean']),
                'Concavity_Mean': float(input_data['concavity_mean']),
                'Concave_Points_Mean': float(input_data['concave_points_mean']),
                'Symmetry_Mean': float(input_data['symmetry_mean']),
                'Fractal_Dimension_Mean': float(input_data['fractal_dimension_mean'])
            }
            
            # Clinical explanations
            if raw_input['Radius_Mean'] > 15:
                explanations.append(f"Enlarged mean cell radius ({raw_input['Radius_Mean']} mm) is indicative of hyperplastic nuclear growth.")
            if raw_input['Concave_Points_Mean'] > 0.05:
                explanations.append(f"High concentration of nuclear concave points ({raw_input['Concave_Points_Mean']}) suggests irregular boundary indentations characteristic of malignancy.")
            if raw_input['Concavity_Mean'] > 0.15:
                explanations.append(f"Elevated nuclear concavity ({raw_input['Concavity_Mean']}) points to structural cell irregularities.")
            if raw_input['Compactness_Mean'] > 0.15:
                explanations.append(f"Elevated cellular compactness ({raw_input['Compactness_Mean']}) indicates cellular crowding within the biopsy structure.")
            if raw_input['Area_Mean'] > 700:
                explanations.append(f"High mean area ({raw_input['Area_Mean']} sq mm) is typical in larger, potentially aggressive tumor cell clusters.")
        
        # Predict
        pred_res = pipeline.predict_single(raw_input, dataset_name, model_name=model_name)
        
        # If probability is high but no positive explanation is triggered, add default
        if pred_res['prediction'] == 1 and len(explanations) == 0:
            explanations.append("Combination of sub-clinical risk features exceeds classifier boundary thresholds.")
        elif pred_res['prediction'] == 0 and len(explanations) == 0:
            explanations.append("All clinical variables fall within normal reference intervals.")
            
        pred_res['explanations'] = explanations
        
        return jsonify({
            'success': True,
            'prediction': pred_res
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5005)
