import urllib.request
import json
import sys

def test_endpoint(url, method='GET', data=None):
    req = urllib.request.Request(url, method=method)
    if data is not None:
        req.add_header('Content-Type', 'application/json')
        json_data = json.dumps(data).encode('utf-8')
        req.data = json_data
    
    try:
        with urllib.request.urlopen(req) as response:
            status = response.status
            body = response.read().decode('utf-8')
            return status, json.loads(body) if 'json' in url or 'api' in url else body
    except Exception as e:
        return 500, str(e)

def run_tests():
    print("Starting AuraMed AI API verification tests...")
    
    # 1. Test Index
    status, body = test_endpoint('http://127.0.0.1:5005/', method='GET')
    print(f"GET /: Status {status}")
    assert status == 200, "Index route failed!"
    assert "AuraMed AI" in body, "Index content mismatch!"
    print("  [PASS] Index Page serves successfully.")
 
    # 2. Test GET Data for Heart
    status, data_res = test_endpoint('http://127.0.0.1:5005/api/data?dataset=heart', method='GET')
    print(f"GET /api/data?dataset=heart: Status {status}")
    assert status == 200, "Heart data API failed!"
    assert data_res['success'] == True, "Data API success field false!"
    assert len(data_res['data']) == 50, f"Expected 50 sample rows, got {len(data_res['data'])}"
    assert 'avg_resting_bp' in data_res['summary_stats'], "Missing avg_resting_bp in summary stats"
    print("  [PASS] Heart data preview fetched successfully.")
 
    # 3. Test GET Data for Diabetes
    status, data_res2 = test_endpoint('http://127.0.0.1:5005/api/data?dataset=diabetes', method='GET')
    print(f"GET /api/data?dataset=diabetes: Status {status}")
    assert status == 200, "Diabetes data API failed!"
    assert data_res2['success'] == True, "Diabetes success field false!"
    assert 'avg_glucose' in data_res2['summary_stats'], "Missing avg_glucose in summary stats"
    print("  [PASS] Diabetes data preview fetched successfully.")
 
    # 4. Test GET Data for Breast Cancer
    status, data_res3 = test_endpoint('http://127.0.0.1:5005/api/data?dataset=breast_cancer', method='GET')
    print(f"GET /api/data?dataset=breast_cancer: Status {status}")
    assert status == 200, "Breast Cancer data API failed!"
    assert data_res3['success'] == True, "Breast Cancer success field false!"
    assert 'avg_radius' in data_res3['summary_stats'], "Missing avg_radius in summary stats"
    print("  [PASS] Breast Cancer data preview fetched successfully.")
 
    # 5. Test POST Train on Heart Dataset
    train_payload = {
        'dataset': 'heart',
        'svm': {'C': 2.0, 'kernel': 'linear'},
        'logistic_regression': {'C': 0.5, 'max_iter': 500},
        'random_forest': {'n_estimators': 50, 'max_depth': 4},
        'xgboost': {'n_estimators': 60, 'learning_rate': 0.15, 'max_depth': 3}
    }
    status, train_res = test_endpoint('http://127.0.0.1:5005/api/train', method='POST', data=train_payload)
    print(f"POST /api/train (Heart): Status {status}")
    assert status == 200, f"Train API failed with {status}!"
    assert train_res['success'] == True, "Train success field false!"
    assert 'svm' in train_res['results'], "Missing SVM results!"
    assert 'xgboost' in train_res['results'], "Missing XGBoost results!"
    print("  [PASS] Model retraining completed successfully.")
 
    # 6. Test Predict Heart Disease (High Risk)
    high_risk_heart = {
        'dataset': 'heart',
        'model_name': 'random_forest',
        'age': 65,
        'sex': 1,
        'cp': 3,
        'trestbps': 160,
        'chol': 280,
        'fbs': 1,
        'restecg': 2,
        'thalach': 110,
        'exang': 1,
        'oldpeak': 3.5,
        'slope': 1,
        'ca': 2,
        'thal': 3
    }
    status, pred_res = test_endpoint('http://127.0.0.1:5005/api/predict', method='POST', data=high_risk_heart)
    print(f"POST /api/predict (Heart - High Risk): Status {status}")
    assert status == 200, "Prediction failed!"
    assert pred_res['success'] == True, "Prediction success field false!"
    prediction = pred_res['prediction']
    print(f"  Decision: {prediction['decision']} (Prob: {prediction['probability']:.1%})")
    assert prediction['prediction'] == 1, "Expected High Risk (1) classification!"
    assert len(prediction['explanations']) > 0, "Expected risk explanations for high risk patient!"
    print("  [PASS] High-risk heart profile assessed correctly.")
 
    # 7. Test Predict Heart Disease (Low Risk)
    low_risk_heart = {
        'dataset': 'heart',
        'model_name': 'random_forest',
        'age': 32,
        'sex': 0,
        'cp': 1,
        'trestbps': 110,
        'chol': 180,
        'fbs': 0,
        'restecg': 0,
        'thalach': 175,
        'exang': 0,
        'oldpeak': 0.0,
        'slope': 2,
        'ca': 0,
        'thal': 2
    }
    status, pred_res2 = test_endpoint('http://127.0.0.1:5005/api/predict', method='POST', data=low_risk_heart)
    print(f"POST /api/predict (Heart - Low Risk): Status {status}")
    assert status == 200, "Prediction failed!"
    assert pred_res2['success'] == True, "Prediction success field false!"
    prediction2 = pred_res2['prediction']
    print(f"  Decision: {prediction2['decision']} (Prob: {prediction2['probability']:.1%})")
    assert prediction2['prediction'] == 0, "Expected Low Risk (0) classification!"
    print("  [PASS] Low-risk heart profile assessed correctly.")

    print("\nAll clinical API endpoints are fully functional! Verification completed successfully.")

if __name__ == '__main__':
    run_tests()
