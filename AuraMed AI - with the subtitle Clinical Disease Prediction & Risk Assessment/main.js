/**
 * AuraMed AI Dashboard Controller
 * Handles tabs, dataset switching, dynamic forms, model training, charting, and prediction.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Global App State
    let currentDataset = 'heart';
    let rocChartInstance = null;
    let importanceChartInstance = null;
    let globalResults = null;

    // Element Selectors
    const datasetSelect = document.getElementById('dataset-select');
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    const trainForm = document.getElementById('train-form');
    const btnTrain = document.getElementById('btn-train');
    const trainSpinner = document.getElementById('train-spinner');
    const trainBtnText = document.getElementById('train-btn-text');

    const importanceModelSelect = document.getElementById('importance-model-select');
    const cmModelSelect = document.getElementById('cm-model-select');

    const predictorForm = document.getElementById('predictor-form');
    const dynamicFormFields = document.getElementById('dynamic-form-fields');
    const predModelSelect = document.getElementById('pred-model');
    const predictionResultPanel = document.getElementById('prediction-result-panel');
    const resultDisplayContent = document.getElementById('result-display-content');
    
    // --- 1. Range Slider Synchronizers ---
    const setupSliders = () => {
        const sliders = [
            { id: 'svm-c', valId: 'svm-c-val' },
            { id: 'lr-c', valId: 'lr-c-val' },
            { id: 'rf-estimators', valId: 'rf-estimators-val' },
            { id: 'rf-depth', valId: 'rf-depth-val' },
            { id: 'xgb-estimators', valId: 'xgb-estimators-val' },
            { id: 'xgb-lr', valId: 'xgb-lr-val' },
            { id: 'xgb-depth', valId: 'xgb-depth-val' }
        ];

        sliders.forEach(slider => {
            const input = document.getElementById(slider.id);
            const output = document.getElementById(slider.valId);
            if (input && output) {
                input.addEventListener('input', () => {
                    output.textContent = input.value;
                });
            }
        });
    };
    setupSliders();

    // --- 2. Theme Toggler ---
    themeToggleBtn.addEventListener('click', () => {
        document.body.classList.toggle('light-theme');
    });

    // --- 3. Navigation Tabs ---
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');

            // Force Chart.js redrawing on tab active if charts exist
            if (tabId === 'overview-tab') {
                if (rocChartInstance) rocChartInstance.resize();
                if (importanceChartInstance) importanceChartInstance.resize();
            }
        });
    });

    // --- 4. Dynamic Prediction Form Generators ---
    const renderPredictorForm = (dataset) => {
        let html = '';
        if (dataset === 'heart') {
            html = `
                <div class="form-input-wrapper">
                    <label for="p-age">Age (years)</label>
                    <input type="number" id="p-age" name="age" min="18" max="95" value="54" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-sex">Biological Sex</label>
                    <select id="p-sex" name="sex" required>
                        <option value="1" selected>Male</option>
                        <option value="0">Female</option>
                    </select>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-cp">Chest Pain Type (cp)</label>
                    <select id="p-cp" name="cp" required>
                        <option value="3" selected>Asymptomatic (Type 3)</option>
                        <option value="2">Non-Anginal Pain (Type 2)</option>
                        <option value="1">Atypical Angina (Type 1)</option>
                        <option value="0">Typical Angina (Type 0)</option>
                    </select>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-trestbps">Resting Blood Pressure (mmHg)</label>
                    <input type="number" id="p-trestbps" name="trestbps" min="80" max="220" value="135" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-chol">Serum Cholesterol (mg/dl)</label>
                    <input type="number" id="p-chol" name="chol" min="100" max="600" value="250" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-fbs">Fasting Blood Sugar > 120 mg/dl</label>
                    <select id="p-fbs" name="fbs" required>
                        <option value="0" selected>False (<= 120 mg/dl)</option>
                        <option value="1">True (> 120 mg/dl)</option>
                    </select>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-restecg">Resting ECG Result</label>
                    <select id="p-restecg" name="restecg" required>
                        <option value="0" selected>Normal (0)</option>
                        <option value="1">ST-T Wave Abnormality (1)</option>
                        <option value="2">Left Ventricular Hypertrophy (2)</option>
                    </select>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-thalach">Max Heart Rate Achieved (thalach)</label>
                    <input type="number" id="p-thalach" name="thalach" min="60" max="220" value="142" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-exang">Exercise Induced Angina</label>
                    <select id="p-exang" name="exang" required>
                        <option value="0" selected>No (0)</option>
                        <option value="1">Yes (1)</option>
                    </select>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-oldpeak">ST Depression (oldpeak)</label>
                    <input type="number" id="p-oldpeak" name="oldpeak" min="0.0" max="8.0" step="0.1" value="1.8" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-slope">Peak Exercise ST Slope</label>
                    <select id="p-slope" name="slope" required>
                        <option value="1" selected>Flat (1)</option>
                        <option value="2">Downsloping (2)</option>
                        <option value="0">Upsloping (0)</option>
                    </select>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-ca">Number of Major Vessels (ca)</label>
                    <input type="number" id="p-ca" name="ca" min="0" max="4" value="1" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-thal">Thalassemia (thal)</label>
                    <select id="p-thal" name="thal" required>
                        <option value="2" selected>Normal / Fixed Defect (2)</option>
                        <option value="3">Reversible Defect (3)</option>
                        <option value="1">Normal Perfusion (1)</option>
                        <option value="0">None (0)</option>
                    </select>
                </div>
            `;
        } else if (dataset === 'diabetes') {
            html = `
                <div class="form-input-wrapper">
                    <label for="p-pregnancies">Pregnancies</label>
                    <input type="number" id="p-pregnancies" name="pregnancies" min="0" max="20" value="2" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-glucose">Plasma Glucose Concentration</label>
                    <input type="number" id="p-glucose" name="glucose" min="40" max="200" value="145" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-bp">Diastolic Blood Pressure (mmHg)</label>
                    <input type="number" id="p-bp" name="blood_pressure" min="40" max="140" value="70" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-skin">Triceps Skin Fold Thickness (mm)</label>
                    <input type="number" id="p-skin" name="skin_thickness" min="0" max="100" value="23" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-insulin">2-Hour Serum Insulin (mIU/L)</label>
                    <input type="number" id="p-insulin" name="insulin" min="0" max="900" value="110" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-bmi">Body Mass Index (BMI)</label>
                    <input type="number" id="p-bmi" name="bmi" min="15.0" max="70.0" step="0.1" value="34.5" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-dpf">Diabetes Pedigree Function (dpf)</label>
                    <input type="number" id="p-dpf" name="dpf" min="0.05" max="2.5" step="0.001" value="0.58" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-age">Age (years)</label>
                    <input type="number" id="p-age" name="age" min="21" max="100" value="48" required>
                </div>
            `;
        } else { // breast_cancer
            html = `
                <div class="form-input-wrapper">
                    <label for="p-radius">Mean Radius (mm)</label>
                    <input type="number" id="p-radius" name="radius_mean" min="5.0" max="35.0" step="0.01" value="17.2" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-texture">Mean Texture</label>
                    <input type="number" id="p-texture" name="texture_mean" min="8.0" max="45.0" step="0.01" value="21.1" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-perimeter">Mean Perimeter (mm)</label>
                    <input type="number" id="p-perimeter" name="perimeter_mean" min="40.0" max="200.0" step="0.1" value="115.0" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-area">Mean Area (sq mm)</label>
                    <input type="number" id="p-area" name="area_mean" min="100.0" max="2600.0" step="1" value="920" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-smoothness">Mean Smoothness</label>
                    <input type="number" id="p-smoothness" name="smoothness_mean" min="0.01" max="0.2" step="0.0001" value="0.108" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-compactness">Mean Compactness</label>
                    <input type="number" id="p-compactness" name="compactness_mean" min="0.01" max="0.4" step="0.001" value="0.18" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-concavity">Mean Concavity</label>
                    <input type="number" id="p-concavity" name="concavity_mean" min="0.0" max="0.5" step="0.001" value="0.21" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-concave">Mean Concave Points</label>
                    <input type="number" id="p-concave" name="concave_points_mean" min="0.0" max="0.25" step="0.001" value="0.095" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-symmetry">Mean Symmetry</label>
                    <input type="number" id="p-symmetry" name="symmetry_mean" min="0.1" max="0.35" step="0.001" value="0.198" required>
                </div>
                <div class="form-input-wrapper">
                    <label for="p-fractal">Mean Fractal Dimension</label>
                    <input type="number" id="p-fractal" name="fractal_dimension_mean" min="0.01" max="0.1" step="0.00001" value="0.065" required>
                </div>
            `;
        }
        dynamicFormFields.innerHTML = html;
        
        // Reset prediction result card to initial empty state
        predictionResultPanel.classList.add('empty');
        resultDisplayContent.classList.add('hidden');
        const placeholder = predictionResultPanel.querySelector('.result-placeholder');
        if (placeholder) placeholder.classList.remove('hidden');
    };

    // --- 5. Data Preview Render ---
    const updateDataPreviewTab = (data, stats) => {
        // Render Summary Stats Cards
        const statsContainer = document.getElementById('data-summary-cards');
        let cardHtml = '';
        
        if (currentDataset === 'heart') {
            cardHtml = `
                <div class="metric-card">
                    <div class="metric-icon"><i class="fa-solid fa-users"></i></div>
                    <div class="metric-info">
                        <h3>Total Patients</h3>
                        <p class="metric-value">${stats.total_patients}</p>
                        <span class="metric-sub">Structured Heart cohort</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon"><i class="fa-solid fa-calendar-days"></i></div>
                    <div class="metric-info">
                        <h3>Average Age</h3>
                        <p class="metric-value">${stats.avg_age.toFixed(1)} yrs</p>
                        <span class="metric-sub">Cleveland mean age</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon"><i class="fa-solid fa-droplet"></i></div>
                    <div class="metric-info">
                        <h3>Mean Resting BP</h3>
                        <p class="metric-value">${stats.avg_resting_bp.toFixed(0)} mmHg</p>
                        <span class="metric-sub">Mean systolic pressure</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon"><i class="fa-solid fa-stethoscope"></i></div>
                    <div class="metric-info">
                        <h3>Heart Disease Rate</h3>
                        <p class="metric-value">${stats.positive_pct.toFixed(1)}%</p>
                        <span class="metric-sub">Positive (Target = 1)</span>
                    </div>
                </div>
            `;
        } else if (currentDataset === 'diabetes') {
            cardHtml = `
                <div class="metric-card">
                    <div class="metric-icon"><i class="fa-solid fa-users"></i></div>
                    <div class="metric-info">
                        <h3>Total Patients</h3>
                        <p class="metric-value">${stats.total_patients}</p>
                        <span class="metric-sub">Pima cohort cohort</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon"><i class="fa-solid fa-gauge-high"></i></div>
                    <div class="metric-info">
                        <h3>Average Glucose</h3>
                        <p class="metric-value">${stats.avg_glucose.toFixed(1)} mg/dL</p>
                        <span class="metric-sub">Plasma concentration</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon"><i class="fa-solid fa-weight-scale"></i></div>
                    <div class="metric-info">
                        <h3>Mean BMI</h3>
                        <p class="metric-value">${stats.avg_bmi.toFixed(1)}</p>
                        <span class="metric-sub">Body Mass Index</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon"><i class="fa-solid fa-kit-medical"></i></div>
                    <div class="metric-info">
                        <h3>Diabetes Rate</h3>
                        <p class="metric-value">${stats.positive_pct.toFixed(1)}%</p>
                        <span class="metric-sub">Positive (Target = 1)</span>
                    </div>
                </div>
            `;
        } else { // breast_cancer
            cardHtml = `
                <div class="metric-card">
                    <div class="metric-icon"><i class="fa-solid fa-users"></i></div>
                    <div class="metric-info">
                        <h3>Total Patients</h3>
                        <p class="metric-value">${stats.total_patients}</p>
                        <span class="metric-sub">Wisconsin Diagnostic samples</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon"><i class="fa-solid fa-maximize"></i></div>
                    <div class="metric-info">
                        <h3>Mean Cell Radius</h3>
                        <p class="metric-value">${stats.avg_radius.toFixed(2)} mm</p>
                        <span class="metric-sub">Nuclear border radius</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon"><i class="fa-solid fa-vector-square"></i></div>
                    <div class="metric-info">
                        <h3>Mean Cell Area</h3>
                        <p class="metric-value">${stats.avg_area.toFixed(1)} mm²</p>
                        <span class="metric-sub">Nuclear footprint size</span>
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-icon"><i class="fa-solid fa-radiation"></i></div>
                    <div class="metric-info">
                        <h3>Malignancy Rate</h3>
                        <p class="metric-value">${stats.positive_pct.toFixed(1)}%</p>
                        <span class="metric-sub">Malignant (Target = 1)</span>
                    </div>
                </div>
            `;
        }
        statsContainer.innerHTML = cardHtml;

        // Render Table Headers
        const thead = document.getElementById('dataset-preview-thead');
        const tbody = document.getElementById('dataset-preview-tbody');
        
        if (!data || data.length === 0) return;
        
        const cols = Object.keys(data[0]);
        thead.innerHTML = `<tr>${cols.map(c => `<th>${c.replace(/_/g, ' ')}</th>`).join('')}</tr>`;

        // Render Table Rows
        tbody.innerHTML = data.map(row => {
            return `<tr>${cols.map(c => {
                let val = row[c];
                if (c === 'Target') {
                    const statusClass = val === 1 ? 'pos' : 'neg';
                    const statusLabel = currentDataset === 'heart' ? (val === 1 ? 'Disease' : 'Normal') :
                                        currentDataset === 'diabetes' ? (val === 1 ? 'Diabetic' : 'Normal') :
                                        (val === 1 ? 'Malignant' : 'Benign');
                    return `<td><span class="badge-target ${statusClass}">${statusLabel} (${val})</span></td>`;
                }
                if (typeof val === 'number') {
                    // Check if integer
                    return val % 1 === 0 ? `<td>${val}</td>` : `<td>${val.toFixed(4)}</td>`;
                }
                return `<td>${val}</td>`;
            }).join('')}</tr>`;
        }).join('');
    };

    // --- 6. Chart.js Visualization logic ---
    const drawCharts = (results) => {
        // Save globally
        globalResults = results;

        // Colors for algorithms
        const colors = {
            svm: { border: '#a855f7', bg: 'rgba(168, 85, 247, 0.1)' },
            logistic_regression: { border: '#ec4899', bg: 'rgba(236, 72, 153, 0.1)' },
            random_forest: { border: '#6366f1', bg: 'rgba(99, 102, 241, 0.1)' },
            xgboost: { border: '#10b981', bg: 'rgba(16, 185, 129, 0.1)' }
        };

        const algoLabels = {
            svm: 'Support Vector Machine',
            logistic_regression: 'Logistic Regression',
            random_forest: 'Random Forest',
            xgboost: 'XGBoost / GBDT'
        };

        // --- ROC Chart ---
        const rocCtx = document.getElementById('roc-chart').getContext('2d');
        if (rocChartInstance) {
            rocChartInstance.destroy();
        }

        const datasetsList = Object.keys(results).map(key => {
            const curve = results[key].roc_curve;
            // Map fpr/tpr to coordinates
            const dataCoords = curve.map(pt => ({ x: pt.fpr, y: pt.tpr }));
            return {
                label: algoLabels[key],
                data: dataCoords,
                borderColor: colors[key].border,
                backgroundColor: colors[key].bg,
                borderWidth: 2.5,
                tension: 0.15,
                fill: false,
                pointRadius: 1,
                pointHoverRadius: 4
            };
        });

        // Add 50% diagonal baseline
        datasetsList.push({
            label: 'Random Guess',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: '#64748b',
            borderDash: [5, 5],
            borderWidth: 1.5,
            fill: false,
            pointRadius: 0
        });

        rocChartInstance = new Chart(rocCtx, {
            type: 'line',
            data: { datasets: datasetsList },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'linear',
                        title: { display: true, text: 'False Positive Rate (1 - Specificity)', color: '#94a3b8' },
                        grid: { color: 'rgba(148, 163, 184, 0.08)' },
                        ticks: { color: '#94a3b8' },
                        min: 0,
                        max: 1
                    },
                    y: {
                        title: { display: true, text: 'True Positive Rate (Sensitivity)', color: '#94a3b8' },
                        grid: { color: 'rgba(148, 163, 184, 0.08)' },
                        ticks: { color: '#94a3b8' },
                        min: 0,
                        max: 1
                    }
                },
                plugins: {
                    legend: {
                        labels: { color: '#94a3b8', font: { family: 'Outfit', size: 10 } }
                    }
                }
            }
        });

        // --- Performance Metrics Table ---
        const tableBody = document.getElementById('metrics-table-body');
        tableBody.innerHTML = Object.keys(results).map(key => {
            const res = results[key];
            return `
                <tr>
                    <td><strong>${algoLabels[key]}</strong></td>
                    <td>${(res.accuracy * 100).toFixed(1)}%</td>
                    <td>${(res.precision * 100).toFixed(1)}%</td>
                    <td>${(res.recall * 100).toFixed(1)}%</td>
                    <td>${(res.f1_score * 100).toFixed(1)}%</td>
                    <td><strong>${res.roc_auc.toFixed(3)}</strong></td>
                </tr>
            `;
        }).join('');

        // --- Metrics Summary Cards ---
        // Find best model based on ROC AUC
        let bestKey = 'random_forest';
        let bestAuc = 0;
        Object.keys(results).forEach(key => {
            if (results[key].roc_auc > bestAuc) {
                bestAuc = results[key].roc_auc;
                bestKey = key;
            }
        });

        document.getElementById('top-model-name').textContent = algoLabels[bestKey];
        document.getElementById('top-model-score').textContent = `ROC AUC: ${bestAuc.toFixed(3)}`;

        // Populate metrics cards with best model metrics
        const bestRes = results[bestKey];
        document.getElementById('avg-accuracy-value').textContent = `${(bestRes.accuracy * 100).toFixed(1)}%`;
        document.getElementById('accuracy-model-name').textContent = `via ${algoLabels[bestKey]}`;
        
        document.getElementById('avg-f1-value').textContent = `${bestRes.f1_score.toFixed(3)}`;
        document.getElementById('f1-model-name').textContent = `via ${algoLabels[bestKey]}`;

        document.getElementById('avg-auc-value').textContent = `${bestRes.roc_auc.toFixed(3)}`;
        document.getElementById('auc-model-name').textContent = `via ${algoLabels[bestKey]}`;

        // Draw feature importances and confusion matrix for selected model
        updateImportanceChart();
        updateConfusionMatrix();
    };

    const updateImportanceChart = () => {
        if (!globalResults) return;
        const selectedModel = importanceModelSelect.value;
        const metrics = globalResults[selectedModel];
        if (!metrics) return;

        const importanceCtx = document.getElementById('importance-chart').getContext('2d');
        if (importanceChartInstance) {
            importanceChartInstance.destroy();
        }

        const featureData = metrics.feature_importances;
        // Sort features by impact value (absolute value for Coefficients/Logistic Regression)
        const sortedFeatures = Object.keys(featureData)
            .map(name => ({ name, value: featureData[name] }))
            .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
            .slice(0, 10); // show top 10 features

        const labels = sortedFeatures.map(item => item.name.replace(/_/g, ' '));
        const values = sortedFeatures.map(item => item.value);

        const isCoeff = selectedModel === 'logistic_regression';
        const chartColor = selectedModel === 'random_forest' ? '#6366f1' :
                           selectedModel === 'xgboost' ? '#10b981' : '#ec4899';

        importanceChartInstance = new Chart(importanceCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: isCoeff ? 'Coefficient Weight' : 'Feature Importance Score',
                    data: values,
                    backgroundColor: chartColor + 'bb',
                    borderColor: chartColor,
                    borderWidth: 1.5,
                    borderRadius: 6
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        grid: { color: 'rgba(148, 163, 184, 0.08)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8', font: { family: 'Outfit', size: 10 } }
                    }
                },
                plugins: {
                    legend: { display: false }
                }
            }
        });
    };

    const updateConfusionMatrix = () => {
        if (!globalResults) return;
        const selectedModel = cmModelSelect.value;
        const metrics = globalResults[selectedModel];
        if (!metrics) return;

        const cm = metrics.confusion_matrix;
        
        document.getElementById('cm-tn').querySelector('.cm-val').textContent = cm.tn;
        document.getElementById('cm-fp').querySelector('.cm-val').textContent = cm.fp;
        document.getElementById('cm-fn').querySelector('.cm-val').textContent = cm.fn;
        document.getElementById('cm-tp').querySelector('.cm-val').textContent = cm.tp;
    };

    importanceModelSelect.addEventListener('change', updateImportanceChart);
    cmModelSelect.addEventListener('change', updateConfusionMatrix);

    // --- 7. Data Loader ---
    const fetchDatasetData = async () => {
        try {
            const response = await fetch(`/api/data?dataset=${currentDataset}`);
            const result = await response.json();
            if (result.success) {
                // Update cohort preview table
                updateDataPreviewTab(result.data, result.summary_stats);
                
                // Redraw comparison charts
                drawCharts(result.default_results);
            } else {
                console.error("API error:", result.error);
            }
        } catch (e) {
            console.error("Network error loading dataset:", e);
        }
    };

    // Handle Active Dataset Selector Switching
    datasetSelect.addEventListener('change', () => {
        currentDataset = datasetSelect.value;
        renderPredictorForm(currentDataset);
        fetchDatasetData();
    });

    // --- 8. Retrain Form Submission ---
    trainForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI loading state
        btnTrain.disabled = true;
        trainSpinner.classList.remove('hidden');
        trainBtnText.textContent = "Retraining Classifiers...";

        // Collect hyperparams
        const payload = {
            dataset: currentDataset,
            svm: {
                C: parseFloat(document.getElementById('svm-c').value),
                kernel: document.getElementById('svm-kernel').value
            },
            logistic_regression: {
                C: parseFloat(document.getElementById('lr-c').value),
                max_iter: parseInt(document.getElementById('lr-max-iter').value)
            },
            random_forest: {
                n_estimators: parseInt(document.getElementById('rf-estimators').value),
                max_depth: parseInt(document.getElementById('rf-depth').value)
            },
            xgboost: {
                n_estimators: parseInt(document.getElementById('xgb-estimators').value),
                learning_rate: parseFloat(document.getElementById('xgb-lr').value),
                max_depth: parseInt(document.getElementById('xgb-depth').value)
            }
        };

        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            if (result.success) {
                drawCharts(result.results);
                
                // Auto switch back to Overview Tab
                tabButtons[0].click();
            } else {
                alert(`Error during retraining: ${result.error}`);
            }
        } catch (err) {
            console.error("Retrain failure:", err);
            alert("Failed to communicate with training API.");
        } finally {
            btnTrain.disabled = false;
            trainSpinner.classList.add('hidden');
            trainBtnText.textContent = "Optimize & Train Models";
        }
    });

    // --- 9. Risk Predictor Logic ---
    predictorForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const btnPredict = document.getElementById('btn-predict');
        btnPredict.disabled = true;
        btnPredict.innerHTML = `<i class="fa-solid fa-spinner spinner-icon"></i> Analysing...`;

        const formData = new FormData(predictorForm);
        const modelName = predModelSelect.value;

        // Reconstruct input payload based on active dataset
        const payload = {
            dataset: currentDataset,
            model_name: modelName
        };

        formData.forEach((value, key) => {
            payload[key] = value;
        });

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const result = await response.json();
            if (result.success) {
                const pred = result.prediction;
                
                // Show Result Panel and Hide Placeholder
                predictionResultPanel.classList.remove('empty');
                const placeholder = predictionResultPanel.querySelector('.result-placeholder');
                if (placeholder) placeholder.classList.add('hidden');
                resultDisplayContent.classList.remove('hidden');

                // 1. Set Risk Decision Badge and color accents
                const badge = document.getElementById('decision-badge');
                const scoreCircle = document.getElementById('score-progress');
                
                badge.className = 'decision-badge';
                
                const posLabel = currentDataset === 'heart' ? 'Heart Disease Pos' :
                                 currentDataset === 'diabetes' ? 'Diabetic Pos' : 'Malignant (Pos)';
                const negLabel = currentDataset === 'heart' ? 'Normal Neg' :
                                 currentDataset === 'diabetes' ? 'Normal Neg' : 'Benign (Neg)';
                
                if (pred.prediction === 1) {
                    badge.textContent = posLabel;
                    badge.classList.add('high-risk');
                    scoreCircle.style.setProperty('--accent-indigo', 'var(--accent-rose)');
                    scoreCircle.querySelector('.score-value').style.color = 'var(--accent-rose)';
                } else {
                    badge.textContent = negLabel;
                    badge.classList.add('low-risk');
                    scoreCircle.style.setProperty('--accent-indigo', 'var(--accent-emerald)');
                    scoreCircle.querySelector('.score-value').style.color = 'var(--accent-emerald)';
                }

                // 2. Animate Circular Progress
                const probPercent = Math.round(pred.probability * 100);
                document.getElementById('score-percentage').textContent = `${probPercent}%`;
                
                // Set conic gradient degree based on percentage
                scoreCircle.style.background = `
                    radial-gradient(closest-side, ${document.body.classList.contains('light-theme') ? '#ffffff' : '#16112c'} 79%, transparent 80% 100%),
                    conic-gradient(var(--accent-indigo) ${probPercent}%, rgba(255,255,255,0.05) 0%)
                `;

                // 3. Set score meta details
                const confidence = pred.prediction === 1 ? pred.probability : (1 - pred.probability);
                document.getElementById('score-probability-text').textContent = `${(confidence * 100).toFixed(2)}%`;
                document.getElementById('score-class-text').textContent = `${pred.prediction === 1 ? posLabel : negLabel} (${pred.prediction})`;

                // 4. Render derived ratios in predictor result panel
                const ratioContainer = document.getElementById('derived-ratios-container');
                let ratioHtml = '';
                
                if (currentDataset === 'heart') {
                    const ratio = parseFloat(payload.trestbps) / parseFloat(payload.chol);
                    const hrAge = parseFloat(payload.thalach) / parseFloat(payload.age);
                    ratioHtml = `
                        <div class="ratio-stat">
                            <span class="ratio-label">BP / Chol. Ratio:</span>
                            <span class="ratio-val">${ratio.toFixed(4)}</span>
                        </div>
                        <div class="ratio-stat">
                            <span class="ratio-label">Max HR / Age:</span>
                            <span class="ratio-val">${hrAge.toFixed(4)}</span>
                        </div>
                    `;
                } else if (currentDataset === 'diabetes') {
                    const insGluc = parseFloat(payload.insulin) / (parseFloat(payload.glucose) + 1.0);
                    const bmiAge = parseFloat(payload.bmi) / parseFloat(payload.age);
                    ratioHtml = `
                        <div class="ratio-stat">
                            <span class="ratio-label">Insulin / Glucose:</span>
                            <span class="ratio-val">${insGluc.toFixed(4)}</span>
                        </div>
                        <div class="ratio-stat">
                            <span class="ratio-label">BMI / Age Ratio:</span>
                            <span class="ratio-val">${bmiAge.toFixed(4)}</span>
                        </div>
                    `;
                } else { // breast_cancer
                    const areaPerim = parseFloat(payload.area_mean) / parseFloat(payload.perimeter_mean);
                    const product = parseFloat(payload.compactness_mean) * parseFloat(payload.concavity_mean);
                    ratioHtml = `
                        <div class="ratio-stat">
                            <span class="ratio-label">Area / Perimeter:</span>
                            <span class="ratio-val">${areaPerim.toFixed(4)}</span>
                        </div>
                        <div class="ratio-stat">
                            <span class="ratio-label">Comp. * Concav.:</span>
                            <span class="ratio-val">${product.toFixed(4)}</span>
                        </div>
                    `;
                }
                ratioContainer.innerHTML = ratioHtml;

                // 5. Populate explanations list
                const factorsList = document.getElementById('risk-factors-list');
                factorsList.innerHTML = pred.explanations.map(exp => `<li>${exp}</li>`).join('');
                
            } else {
                alert(`Prediction error: ${result.error}`);
            }
        } catch (err) {
            console.error("Prediction failure:", err);
            alert("Failed to connect to evaluation API.");
        } finally {
            btnPredict.disabled = false;
            btnPredict.innerHTML = `<i class="fa-solid fa-heart-pulse"></i> Evaluate Patient Risk`;
        }
    });

    // Initialize application on startup
    renderPredictorForm('heart');
    fetchDatasetData();
});
