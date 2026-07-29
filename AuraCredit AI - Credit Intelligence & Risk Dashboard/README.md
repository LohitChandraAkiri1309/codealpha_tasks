# AuraCredit AI - Credit Intelligence & Risk Dashboard

AuraCredit AI is an intelligent credit risk assessment platform and dashboard. It uses synthetic financial data to train and compare multiple Machine Learning classification models on startup, allowing users to assess simulated loan applicants in real-time.

---

## Features

- **Synthetic Data Generator**: Generates realistic applicant profiles with attributes like age, income, employment tenure, home ownership, credit utilization, and payment histories.
- **Model Training & Comparison**: Automatically trains and evaluates **Logistic Regression**, **Decision Tree**, and **Random Forest** classifiers.
- **Interactive Dashboard**: View dataset summaries, statistics, and predict creditworthiness on user-specified applicant parameters.
- **Explainable AI**: Provides immediate clinical/financial explanations for predictions (e.g., impact of high debt-to-income ratio or delinquencies).

---

## How to Setup and Run

### 1. Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 2. Open the Project Folder
Open your terminal or command prompt and navigate to the project directory:
```bash
cd "AuraCredit AI - Credit Intelligence & Risk Dashboard"
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

Upon starting, the application will automatically generate synthetic data, pre-train the classification models, and host the web page locally.

### 6. View the Dashboard
Open your web browser and navigate to:
```
http://127.0.0.1:5000/
```

> [!WARNING]
> **Port Conflict:** By default, this application runs on port `5000`. If you plan to run **Neural Characters** concurrently, you will need to change the port number in `app.py` for one of the projects to avoid a port collision.

---

## Technical Details

- **Backend**: Python Flask framework.
- **Data Handling**: Pandas & NumPy for data parsing and pre-processing.
- **Machine Learning**: Scikit-Learn classifiers (`LogisticRegression`, `DecisionTreeClassifier`, `RandomForestClassifier`).
- **Web UI**: Vanilla HTML/JS front-end communicating with Flask API endpoints.
