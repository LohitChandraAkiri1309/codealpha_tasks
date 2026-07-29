// AuraCredit AI Javascript Logic Controller

document.addEventListener('DOMContentLoaded', () => {
    // Chart References
    let rocChart = null;
    let metricsChart = null;
    let importanceChart = null;

    // Cache metrics data to redraw importance chart dynamically
    let cachedResults = null;

    // DOM Elements
    const themeToggleBtn = document.getElementById('theme-toggle-btn');
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    // Sliders & Values
    const sliders = [
        { id: 'lr-c', valId: 'lr-c-val' },
        { id: 'dt-depth', valId: 'dt-depth-val' },
        { id: 'dt-split', valId: 'dt-split-val' },
        { id: 'rf-estimators', valId: 'rf-estimators-val' },
        { id: 'rf-depth', valId: 'rf-depth-val' }
    ];

    // Train Form
    const trainForm = document.getElementById('train-form');
    const btnTrain = document.getElementById('btn-train');
    const trainSpinner = document.getElementById('train-spinner');
    const trainBtnText = document.getElementById('train-btn-text');

    // Predict Form
    const predictionForm = document.getElementById('prediction-form');
    const btnPredict = document.getElementById('btn-predict');
    const predictionResultPanel = document.getElementById('prediction-result-panel');
    const resultPlaceholder = predictionResultPanel.querySelector('.result-placeholder');
    const resultDisplayContent = document.getElementById('result-display-content');
    
    // Importance selector
    const importanceModelSelect = document.getElementById('importance-model-select');

    /* ==========================================
       1. Theme Management
       ========================================== */
    const loadTheme = () => {
        const savedTheme = localStorage.getItem('theme') || 'dark';
        if (savedTheme === 'light') {
            document.body.classList.remove('dark-theme');
            document.body.classList.add('light-theme');
        } else {
            document.body.classList.add('dark-theme');
            document.body.classList.remove('light-theme');
        }
    };

    themeToggleBtn.addEventListener('click', () => {
        if (document.body.classList.contains('light-theme')) {
            document.body.classList.remove('light-theme');
            document.body.classList.add('dark-theme');
            localStorage.setItem('theme', 'dark');
        } else {
            document.body.classList.remove('dark-theme');
            document.body.classList.add('light-theme');
            localStorage.setItem('theme', 'light');
        }
        // Redraw charts to update text colors in new theme
        if (cachedResults) {
            updateCharts(cachedResults);
        }
    });

    loadTheme();

    /* ==========================================
       2. Navigation Tabs
       ========================================== */
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetPaneId = btn.getAttribute('data-tab');
            
            // Toggle buttons
            tabButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Toggle panes
            tabPanes.forEach(pane => {
                if (pane.id === targetPaneId) {
                    pane.classList.add('active');
                } else {
                    pane.classList.remove('active');
                }
            });
        });
    });

    /* ==========================================
       3. Slider Synchronization
       ========================================== */
    sliders.forEach(slider => {
        const el = document.getElementById(slider.id);
        const displayEl = document.getElementById(slider.valId);
        if (el && displayEl) {
            el.addEventListener('input', (e) => {
                displayEl.textContent = e.target.value;
            });
        }
    });

    /* ==========================================
       4. Fetch Dataset & Stats
       ========================================== */
    const loadDataset = async () => {
        try {
            const response = await fetch('/api/data');
            const resData = await response.json();
            
            if (resData.success) {
                // Populate dataset summary metrics
                const stats = resData.summary_stats;
                document.getElementById('ds-size').textContent = `${stats.total_applicants} Applicants`;
                document.getElementById('ds-avg-income').textContent = `$${Math.round(stats.avg_income).toLocaleString()}`;
                document.getElementById('ds-avg-util').textContent = `${(stats.avg_utilization * 100).toFixed(1)}%`;
                document.getElementById('ds-avg-dti').textContent = `${(stats.avg_dti * 100).toFixed(1)}%`;
                document.getElementById('ds-cw-ratio').textContent = `${stats.creditworthy_pct.toFixed(1)}%`;

                // Populate applicants table
                const tableBody = document.querySelector('#applicants-table tbody');
                tableBody.innerHTML = ''; // Clear previous

                resData.data.forEach(row => {
                    const tr = document.createElement('tr');
                    
                    const utilization = (row.Credit_Utilization_Ratio * 100).toFixed(1);
                    const dti = (row.Debt_to_Income_Ratio * 100).toFixed(1);
                    
                    const statusBadge = row.Creditworthy === 1 
                        ? '<span class="badge badge-success">Creditworthy</span>'
                        : '<span class="badge badge-danger">High Risk</span>';

                    tr.innerHTML = `
                        <td>${row.Age}</td>
                        <td>$${row.Annual_Income.toLocaleString()}</td>
                        <td>${row.Home_Ownership}</td>
                        <td>$${row.Credit_Limit.toLocaleString()}</td>
                        <td>$${row.Current_Balance.toLocaleString()}</td>
                        <td>$${row.Monthly_Debt_Payments.toLocaleString()}</td>
                        <td>${row.Payment_History_Delinquencies}</td>
                        <td class="font-semibold">${utilization}%</td>
                        <td class="font-semibold">${dti}%</td>
                        <td>${statusBadge}</td>
                    `;
                    tableBody.appendChild(tr);
                });
            } else {
                console.error("Failed to load dataset:", resData.error);
            }
        } catch (error) {
            console.error("Network error fetching dataset:", error);
        }
    };

    /* ==========================================
       5. Model Training & Charts Render
       ========================================== */
    const trainModels = async (e) => {
        if (e) e.preventDefault();
        
        // Show spinner
        trainSpinner.classList.remove('hidden');
        trainBtnText.textContent = "Training models...";
        btnTrain.disabled = true;

        // Gather hyperparameters
        const body = {
            logistic_regression: {
                C: parseFloat(document.getElementById('lr-c').value),
                max_iter: parseInt(document.getElementById('lr-max-iter').value)
            },
            decision_tree: {
                max_depth: parseInt(document.getElementById('dt-depth').value),
                min_samples_split: parseInt(document.getElementById('dt-split').value)
            },
            random_forest: {
                n_estimators: parseInt(document.getElementById('rf-estimators').value),
                max_depth: parseInt(document.getElementById('rf-depth').value)
            }
        };

        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });
            const resData = await response.json();

            if (resData.success) {
                cachedResults = resData.results;
                updateDashboardMetrics(cachedResults);
                updateCharts(cachedResults);
            } else {
                alert("Error training models: " + resData.error);
            }
        } catch (error) {
            console.error("Network error during training:", error);
            alert("Network error training models.");
        } finally {
            // Hide spinner
            trainSpinner.classList.add('hidden');
            trainBtnText.textContent = "Train Classification Models";
            btnTrain.disabled = false;
        }
    };

    const updateDashboardMetrics = (results) => {
        // Update AUC indicators
        document.getElementById('rf-auc-display').textContent = results.random_forest.roc_auc.toFixed(4);
        document.getElementById('dt-auc-display').textContent = results.decision_tree.roc_auc.toFixed(4);
        document.getElementById('lr-auc-display').textContent = results.logistic_regression.roc_auc.toFixed(4);

        // Calculate champion model
        let champion = 'random_forest';
        let maxAuc = results.random_forest.roc_auc;
        
        if (results.decision_tree.roc_auc > maxAuc) {
            champion = 'decision_tree';
            maxAuc = results.decision_tree.roc_auc;
        }
        if (results.logistic_regression.roc_auc > maxAuc) {
            champion = 'logistic_regression';
        }

        const championNames = {
            'random_forest': 'Random Forest',
            'decision_tree': 'Decision Tree',
            'logistic_regression': 'Logistic Regression'
        };
        document.getElementById('champion-model-name').textContent = championNames[champion];

        // Populate Confusion Matrices
        const models = ['random_forest', 'decision_tree', 'logistic_regression'];
        const prefix = {
            'random_forest': 'rf',
            'decision_tree': 'dt',
            'logistic_regression': 'lr'
        };

        models.forEach(model => {
            const cm = results[model].confusion_matrix;
            const p = prefix[model];
            document.getElementById(`${p}-cm-tn`).textContent = cm.tn;
            document.getElementById(`${p}-cm-fp`).textContent = cm.fp;
            document.getElementById(`${p}-cm-fn`).textContent = cm.fn;
            document.getElementById(`${p}-cm-tp`).textContent = cm.tp;
        });
    };

    const getThemeColor = (variableName) => {
        return getComputedStyle(document.body).getPropertyValue(variableName).trim();
    };

    const updateCharts = (results) => {
        const textPrimary = getThemeColor('--text-primary');
        const borderColor = getThemeColor('--border-color');
        
        // 1. Chart - Metrics Comparison
        if (metricsChart) metricsChart.destroy();
        
        const models = ['random_forest', 'decision_tree', 'logistic_regression'];
        const modelLabels = ['Random Forest', 'Decision Tree', 'Logistic Regression'];
        const modelColors = [getThemeColor('--rf-color'), getThemeColor('--dt-color'), getThemeColor('--lr-color')];
        
        const metrics = ['accuracy', 'precision', 'recall', 'f1_score'];
        const metricLabels = ['Accuracy', 'Precision', 'Recall', 'F1-Score'];
        
        const datasets = models.map((model, idx) => ({
            label: modelLabels[idx],
            data: metrics.map(m => results[model][m]),
            backgroundColor: modelColors[idx],
            borderRadius: 6
        }));

        const ctxMetrics = document.getElementById('metricsChart').getContext('2d');
        metricsChart = new Chart(ctxMetrics, {
            type: 'bar',
            data: {
                labels: metricLabels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: textPrimary } }
                },
                scales: {
                    x: { 
                        grid: { color: borderColor },
                        ticks: { color: textPrimary }
                    },
                    y: { 
                        min: 0.5, 
                        max: 1.0, 
                        grid: { color: borderColor },
                        ticks: { color: textPrimary }
                    }
                }
            }
        });

        // 2. Chart - ROC Curves
        if (rocChart) rocChart.destroy();

        const rocDatasets = models.map((model, idx) => ({
            label: `${modelLabels[idx]} (AUC: ${results[model].roc_auc.toFixed(3)})`,
            data: results[model].roc_curve.map(pt => ({ x: pt.fpr, y: pt.tpr })),
            borderColor: modelColors[idx],
            backgroundColor: 'transparent',
            borderWidth: 2,
            tension: 0.1,
            pointRadius: 1
        }));

        // Add Random line
        rocDatasets.push({
            label: 'Random Chance',
            data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
            borderColor: getThemeColor('--text-muted'),
            borderDash: [5, 5],
            borderWidth: 1.5,
            pointRadius: 0,
            backgroundColor: 'transparent'
        });

        const ctxRoc = document.getElementById('rocChart').getContext('2d');
        rocChart = new Chart(ctxRoc, {
            type: 'line',
            data: { datasets: rocDatasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: textPrimary } }
                },
                scales: {
                    x: {
                        type: 'linear',
                        min: 0,
                        max: 1,
                        title: { display: true, text: 'False Positive Rate', color: textPrimary },
                        grid: { color: borderColor },
                        ticks: { color: textPrimary }
                    },
                    y: {
                        min: 0,
                        max: 1,
                        title: { display: true, text: 'True Positive Rate', color: textPrimary },
                        grid: { color: borderColor },
                        ticks: { color: textPrimary }
                    }
                }
            }
        });

        // 3. Chart - Feature Importances
        drawImportanceChart();
    };

    const drawImportanceChart = () => {
        if (!cachedResults) return;

        const textPrimary = getThemeColor('--text-primary');
        const borderColor = getThemeColor('--border-color');

        const selectedModel = importanceModelSelect.value;
        const impData = cachedResults[selectedModel].feature_importances;
        
        // Sort features by importance
        const sortedFeatures = Object.entries(impData)
            .sort((a, b) => Math.abs(a[1]) - Math.abs(b[1])); // Ascending order for horizontal bar
            
        const labels = sortedFeatures.map(item => {
            // Make labels user friendly
            return item[0].replace(/_/g, ' ');
        });
        const values = sortedFeatures.map(item => item[1]);

        if (importanceChart) importanceChart.destroy();

        // Color based on sign for Logistic Regression coefficients, otherwise standard model color
        const baseColor = selectedModel === 'random_forest' ? getThemeColor('--rf-color') 
                       : selectedModel === 'decision_tree' ? getThemeColor('--dt-color') 
                       : getThemeColor('--lr-color');
                       
        const backgroundColors = values.map(val => {
            if (selectedModel === 'logistic_regression') {
                return val >= 0 ? getThemeColor('--success') : getThemeColor('--danger');
            }
            return baseColor;
        });

        const ctxImportance = document.getElementById('importanceChart').getContext('2d');
        importanceChart = new Chart(ctxImportance, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: selectedModel === 'logistic_regression' ? 'Coefficients (Impact Direction)' : 'Feature Importance',
                    data: values,
                    backgroundColor: backgroundColors,
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: borderColor },
                        ticks: { color: textPrimary }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: textPrimary }
                    }
                }
            }
        });
    };

    importanceModelSelect.addEventListener('change', drawImportanceChart);

    /* ==========================================
       6. Applicant Creditworthiness Predictor
       ========================================== */
    predictionForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Gather input values
        const payload = {
            age: parseInt(document.getElementById('pred-age').value),
            annual_income: parseFloat(document.getElementById('pred-income').value),
            home_ownership: document.getElementById('pred-home').value,
            credit_limit: parseFloat(document.getElementById('pred-limit').value),
            current_balance: parseFloat(document.getElementById('pred-balance').value),
            monthly_debt: parseFloat(document.getElementById('pred-debt').value),
            employment_years: parseInt(document.getElementById('pred-emp').value),
            delinquencies: parseInt(document.getElementById('pred-delinquencies').value),
            model_name: document.getElementById('pred-model').value
        };

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const resData = await response.json();

            if (resData.success) {
                const pred = resData.prediction;
                
                // Show output card and hide placeholder
                resultPlaceholder.classList.add('hidden');
                resultDisplayContent.classList.remove('hidden');
                predictionResultPanel.classList.remove('empty');

                // Render Decision Badge
                const badge = document.getElementById('decision-badge');
                badge.className = 'decision-badge'; // Reset classes
                if (pred.prediction === 1) {
                    badge.textContent = 'Approved';
                    badge.classList.add('approved');
                } else {
                    badge.textContent = 'Rejected';
                    badge.classList.add('rejected');
                }

                // Render Radial circular score
                const percentScore = Math.round(pred.probability * 100);
                document.getElementById('score-percentage').textContent = `${percentScore}%`;
                document.getElementById('score-probability-text').textContent = `${(pred.probability * 100).toFixed(1)}%`;
                document.getElementById('score-class-text').textContent = pred.prediction === 1 ? 'Creditworthy (1)' : 'High Risk (0)';

                const progressRing = document.getElementById('score-progress');
                const progressColor = pred.prediction === 1 ? getThemeColor('--success') : getThemeColor('--danger');
                const trackColor = document.body.classList.contains('light-theme') ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.05)';
                progressRing.style.background = `conic-gradient(${progressColor} 0% ${percentScore}%, ${trackColor} ${percentScore}% 100%)`;

                // Render derived values
                const utilRatio = payload.credit_limit > 0 ? (payload.current_balance / payload.credit_limit) * 100 : 0;
                const dtiRatio = (payload.monthly_debt * 12 / payload.annual_income) * 100;
                const ptiRatio = (payload.monthly_debt / (payload.annual_income / 12)) * 100;

                document.getElementById('res-util').textContent = `${utilRatio.toFixed(1)}%`;
                document.getElementById('res-dti').textContent = `${dtiRatio.toFixed(1)}%`;
                document.getElementById('res-pti').textContent = `${ptiRatio.toFixed(1)}%`;

                // Render explanations
                const list = document.getElementById('risk-factors-list');
                list.innerHTML = '';
                
                if (pred.explanations && pred.explanations.length > 0) {
                    pred.explanations.forEach(exp => {
                        const li = document.createElement('li');
                        li.textContent = exp;
                        list.appendChild(li);
                    });
                } else {
                    const li = document.createElement('li');
                    li.textContent = "No significant risk flags detected for this applicant.";
                    list.appendChild(li);
                }
            } else {
                alert("Prediction Error: " + resData.error);
            }
        } catch (error) {
            console.error("Network error during prediction:", error);
            alert("Network error assessing creditworthiness.");
        }
    });

    /* ==========================================
       7. Initial Startup Execution
       ========================================== */
    // Load dataset data table
    loadDataset();
    
    // Train models immediately on startup to get metrics
    trainModels();
});
