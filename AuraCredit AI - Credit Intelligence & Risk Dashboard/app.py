# pyrefly: ignore [missing-import]
from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import pandas as pd
from model_pipeline import CreditScoringPipeline

app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialize pipeline and data
pipeline = CreditScoringPipeline()
data_df = pipeline.generate_synthetic_data(num_samples=1000)

# Pre-train models on startup so the app is immediately functional
print("Pre-training models on startup...")
training_results = pipeline.train_and_evaluate(data_df)
pipeline.save_pipeline()
print("Startup models trained and saved successfully.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data', methods=['GET'])
def get_data():
    try:
        # Convert first 50 rows to dictionary for display
        sample_df = data_df.head(50).copy()
        
        # Calculate engineered features for display
        sample_df['Credit_Utilization_Ratio'] = (sample_df['Current_Balance'] / sample_df['Credit_Limit']).round(4)
        sample_df['Debt_to_Income_Ratio'] = ((sample_df['Monthly_Debt_Payments'] * 12) / sample_df['Annual_Income']).round(4)
        sample_df['Payment_to_Income_Ratio'] = (sample_df['Monthly_Debt_Payments'] / (sample_df['Annual_Income'] / 12)).round(4)
        
        records = sample_df.to_dict(orient='records')
        
        # Calculate summary statistics for the dataset
        stats = {
            'total_applicants': len(data_df),
            'avg_income': float(data_df['Annual_Income'].mean()),
            'avg_utilization': float((data_df['Current_Balance'] / data_df['Credit_Limit']).mean()),
            'avg_dti': float(((data_df['Monthly_Debt_Payments'] * 12) / data_df['Annual_Income']).mean()),
            'creditworthy_pct': float((data_df['Creditworthy'].sum() / len(data_df)) * 100)
        }
        
        return jsonify({
            'success': True,
            'data': records,
            'summary_stats': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_models():
    try:
        params = request.get_json() or {}
        
        # Parse parameters
        lr_params = params.get('logistic_regression', {})
        dt_params = params.get('decision_tree', {})
        rf_params = params.get('random_forest', {})
        
        # Filter parameters to ensure valid types
        lr_args = {
            'C': float(lr_params.get('C', 1.0)),
            'max_iter': int(lr_params.get('max_iter', 1000)),
            'random_state': 42
        }
        
        dt_args = {
            'max_depth': int(dt_params.get('max_depth', 5)) if dt_params.get('max_depth') else None,
            'min_samples_split': int(dt_params.get('min_samples_split', 10)),
            'random_state': 42
        }
        
        rf_args = {
            'n_estimators': int(rf_params.get('n_estimators', 100)),
            'max_depth': int(rf_params.get('max_depth', 6)) if rf_params.get('max_depth') else None,
            'random_state': 42
        }
        
        # Retrain
        results = pipeline.train_and_evaluate(
            data_df, 
            lr_params=lr_args, 
            dt_params=dt_args, 
            rf_params=rf_args
        )
        
        # Save pipeline
        pipeline.save_pipeline()
        
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
            
        model_name = input_data.get('model_name', 'random_forest')
        
        # Extract features and convert to floats/integers
        raw_input = {
            'Age': int(input_data['age']),
            'Annual_Income': float(input_data['annual_income']),
            'Credit_Limit': float(input_data['credit_limit']),
            'Current_Balance': float(input_data['current_balance']),
            'Monthly_Debt_Payments': float(input_data['monthly_debt']),
            'Employment_Duration_Years': int(input_data['employment_years']),
            'Payment_History_Delinquencies': int(input_data['delinquencies']),
            'Home_Ownership': input_data['home_ownership']
        }
        
        # Run prediction
        pred_res = pipeline.predict_single(raw_input, model_name=model_name)
        
        # Provide explanation factors
        explanation = []
        dti = (raw_input['Monthly_Debt_Payments'] * 12) / raw_input['Annual_Income']
        util = raw_input['Current_Balance'] / raw_input['Credit_Limit'] if raw_input['Credit_Limit'] > 0 else 0
        
        if raw_input['Payment_History_Delinquencies'] > 2:
            explanation.append("Frequent delinquent payments significantly lower the creditworthiness score.")
        if util > 0.6:
            explanation.append(f"High credit utilization ({util:.1%}) indicates elevated outstanding balance risk.")
        elif util < 0.3 and util > 0:
            explanation.append(f"Healthy credit utilization ({util:.1%}) supports low risk profile.")
            
        if dti > 0.4:
            explanation.append(f"High debt-to-income ratio ({dti:.1%}) impacts monthly repayment capacity.")
        elif dti < 0.2:
            explanation.append(f"Low debt-to-income ratio ({dti:.1%}) indicates strong financial margin.")
            
        if raw_input['Annual_Income'] < 30000:
            explanation.append("Low annual income limits borrowing safety buffers.")
        elif raw_input['Annual_Income'] > 100000:
            explanation.append("High annual income provides robust repayment capacity.")
            
        if raw_input['Employment_Duration_Years'] >= 5:
            explanation.append(f"Stable employment tenure ({raw_input['Employment_Duration_Years']} years) reduces default risk.")
            
        pred_res['explanations'] = explanation
        
        return jsonify({
            'success': True,
            'prediction': pred_res
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
