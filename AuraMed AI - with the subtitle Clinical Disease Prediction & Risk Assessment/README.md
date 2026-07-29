# AuraMed AI - Clinical Disease Prediction & Risk Assessment

AuraMed AI is a clinical decision support and health risk assessment platform. Using synthetic medical datasets, it trains and evaluates several machine learning classifiers to predict risk categories for three common health conditions: Heart Disease, Diabetes, and Breast Cancer. It also supports automated clinical PDF report generation.

---

## Features

- **Multi-Disease Dashboards**: Assess patients across three standard clinical conditions:
  - **Heart Disease**: Predicts coronary risk using features like age, resting blood pressure, cholesterol, max heart rate, and vessel occlusion.
  - **Diabetes**: Predicts diabetic onset probability using indicators like glucose level, blood pressure, BMI, age, and insulin response.
  - **Breast Cancer**: Predicts malignancy using cellular characteristics such as cell radius, concavity, and compactness.
- **Model Suite**: Automatically trains **Support Vector Machine (SVM)**, **Logistic Regression**, **Random Forest**, and **XGBoost** (falls back to Scikit-Learn Gradient Boosting if XGBoost is not installed).
- **PDF Report Generation**: Creates stylized clinical evaluation summary PDF reports containing prediction outcomes, risk probabilities, and detailed feature breakdowns.
- **Clinical Explanations**: Generates professional, medically grounded explanations outlining why an applicant received a high or low risk prediction.

---

## How to Setup and Run

### 1. Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 2. Open the Project Folder
Open your terminal or command prompt and navigate to the project directory:
```bash
cd "AuraMed AI - with the subtitle Clinical Disease Prediction & Risk Assessment"
```

### 3. Create a Virtual Environment (Recommended)
It is highly recommended to isolate your dependencies using a Python virtual environment:
- **Windows**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### 4. Install Dependencies
Install all the required Python libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 5. Run the Application
Start the Flask development server:
```bash
python app.py
```

Upon starting, the application will generate synthetic datasets for all three diseases, train the four classifiers, and save the pipelines.

### 6. View the Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5005/
```

---

## Technical Details

- **Backend**: Python Flask framework.
- **PDF Engine**: ReportLab to compile clean, clinical-grade medical reports.
- **Models**:
  - `SVC` (Support Vector Classifier)
  - `LogisticRegression`
  - `RandomForestClassifier`
  - `XGBClassifier` (or fallback `GradientBoostingClassifier`)
- **Data Engineering**: Dynamic patient feature calculations (e.g., BP to Cholesterol ratio, compact-to-concavity product) to improve classifier decision boundaries.
