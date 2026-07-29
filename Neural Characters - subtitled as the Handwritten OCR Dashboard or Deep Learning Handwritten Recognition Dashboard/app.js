document.addEventListener('DOMContentLoaded', () => {
    // Model Status Check
    checkModelStatus();
    setInterval(checkModelStatus, 5000); // Check status every 5s

    // DOM Elements
    const canvas = document.getElementById('paint-canvas');
    const ctx = canvas.getContext('2d');
    const clearBtn = document.getElementById('clear-btn');
    const undoBtn = document.getElementById('undo-btn');
    const brushSizeSlider = document.getElementById('brush-size');
    const brushSizeVal = document.getElementById('brush-size-val');
    
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const previewWrapper = document.querySelector('.preview-wrapper');
    const uploadPreview = document.getElementById('upload-preview');
    const changeUploadBtn = document.getElementById('change-upload-btn');
    
    const modeBtnSingle = document.getElementById('mode-single');
    const modeBtnSequence = document.getElementById('mode-sequence');
    
    const singleResultView = document.getElementById('single-result-view');
    const sequenceResultView = document.getElementById('sequence-result-view');
    const resultLoader = document.getElementById('result-loader');
    
    const predictedValue = document.getElementById('predicted-value');
    const predictedConfidence = document.getElementById('predicted-confidence');
    const speakBtn = document.getElementById('speak-btn');
    const previewCanvas = document.getElementById('preview-canvas');
    const pCtx = previewCanvas.getContext('2d');
    const distributionBars = document.getElementById('distribution-bars');
    
    const sequenceValue = document.getElementById('sequence-value');
    const speakSequenceBtn = document.getElementById('speak-sequence-btn');
    const annotatedSequencePreview = document.getElementById('annotated-sequence-preview');
    const sequenceImagePlaceholder = document.getElementById('sequence-image-placeholder');
    const segmentedListContainer = document.getElementById('segmented-list-container');
    const layerOutputDims = document.getElementById('layer-output-dims');

    // Drawing Constants
    let isDrawing = false;
    let strokeHistory = [];
    let predictionTimeout = null;
    let currentInputMode = 'draw'; // 'draw' or 'upload'
    let currentInferenceMode = 'single'; // 'single' or 'sequence'
    let selectedModel = 'mnist'; // 'mnist' or 'emnist'

    // Initialize Canvas Context
    initCanvas();

    function initCanvas() {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        ctx.strokeStyle = '#ffffff';
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.lineWidth = brushSizeSlider.value;
    }

    // Brush Size Adjustments
    brushSizeSlider.addEventListener('input', (e) => {
        const val = e.target.value;
        brushSizeVal.textContent = `${val}px`;
        ctx.lineWidth = val;
    });

    // Drawing Mechanics (Mouse)
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    window.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseleave', () => { isDrawing = false; });

    // Drawing Mechanics (Touch)
    canvas.addEventListener('touchstart', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        const rect = canvas.getBoundingClientRect();
        startDrawing({
            clientX: touch.clientX,
            clientY: touch.clientY
        });
    });
    canvas.addEventListener('touchmove', (e) => {
        e.preventDefault();
        const touch = e.touches[0];
        draw({
            clientX: touch.clientX,
            clientY: touch.clientY
        });
    });
    window.addEventListener('touchend', stopDrawing);

    function getMousePos(e) {
        const rect = canvas.getBoundingClientRect();
        // Scale factor due to CSS scaling
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        return {
            x: (e.clientX - rect.left) * scaleX,
            y: (e.clientY - rect.top) * scaleY
        };
    }

    function startDrawing(e) {
        isDrawing = true;
        
        // Push state for undo
        if (strokeHistory.length >= 25) {
            strokeHistory.shift(); // Limit history size
        }
        strokeHistory.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
        
        const pos = getMousePos(e);
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
    }

    function draw(e) {
        if (!isDrawing) return;
        const pos = getMousePos(e);
        ctx.lineTo(pos.x, pos.y);
        ctx.stroke();
    }

    function stopDrawing() {
        if (!isDrawing) return;
        isDrawing = false;
        ctx.beginPath();
        // Debounce auto prediction when user finishes drawing
        debouncePrediction();
    }

    // Canvas Control Buttons
    clearBtn.addEventListener('click', clearCanvas);
    undoBtn.addEventListener('click', undo);

    function clearCanvas() {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        strokeHistory = [];
        resetOutputs();
    }

    function undo() {
        if (strokeHistory.length > 0) {
            const lastState = strokeHistory.pop();
            ctx.putImageData(lastState, 0, 0);
            debouncePrediction(50); // Fast trigger on undo
        }
    }

    // Tab Switching Logic
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
            
            currentInputMode = tabId === 'draw-tab' ? 'draw' : 'upload';
            resetOutputs();
            
            if (currentInputMode === 'upload' && uploadPreview.src) {
                runInference();
            } else if (currentInputMode === 'draw') {
                debouncePrediction(50);
            }
        });
    });

    // File Upload / Dropzone Mechanics
    dropzone.addEventListener('click', () => fileInput.click());
    
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });
    
    dropzone.addEventListener('dragleave', () => {
        dropzone.classList.remove('dragover');
    });
    
    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleUploadedFile(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleUploadedFile(e.target.files[0]);
        }
    });

    changeUploadBtn.addEventListener('click', () => {
        fileInput.value = '';
        uploadPreview.src = '';
        previewWrapper.style.display = 'none';
        dropzone.style.display = 'flex';
        resetOutputs();
    });

    function handleUploadedFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please select an image file.');
            return;
        }
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreview.src = e.target.result;
            dropzone.style.display = 'none';
            previewWrapper.style.display = 'flex';
            runInference();
        };
        reader.readAsDataURL(file);
    }

    // Model Radios & Inference Modes
    document.querySelectorAll('input[name="model-select"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            selectedModel = e.target.value;
            // Update architecture output layer label dimensions
            if (selectedModel === 'mnist') {
                layerOutputDims.textContent = '10 logits';
            } else {
                layerOutputDims.textContent = '26 logits';
            }
            runInference();
        });
    });

    modeBtnSingle.addEventListener('click', () => {
        if (currentInferenceMode === 'single') return;
        currentInferenceMode = 'single';
        modeBtnSingle.classList.add('active');
        modeBtnSequence.classList.remove('active');
        
        singleResultView.style.display = 'flex';
        sequenceResultView.style.display = 'none';
        runInference();
    });

    modeBtnSequence.addEventListener('click', () => {
        if (currentInferenceMode === 'sequence') return;
        currentInferenceMode = 'sequence';
        modeBtnSequence.classList.add('active');
        modeBtnSingle.classList.remove('active');
        
        singleResultView.style.display = 'none';
        sequenceResultView.style.display = 'flex';
        runInference();
    });

    // Debouncing utility
    function debouncePrediction(ms = 400) {
        if (predictionTimeout) clearTimeout(predictionTimeout);
        predictionTimeout = setTimeout(() => {
            runInference();
        }, ms);
    }

    // Run Inference pipeline
    function runInference() {
        let imageData = '';
        
        if (currentInputMode === 'draw') {
            // Check if canvas is blank/empty by testing pixel buffer
            const buffer = ctx.getImageData(0, 0, canvas.width, canvas.height);
            const isBlank = !buffer.data.some((val, i) => i % 4 !== 3 && val > 0); // Ignore alpha channel
            if (isBlank) {
                resetOutputs();
                return;
            }
            imageData = canvas.toDataURL('image/png');
        } else {
            if (!uploadPreview.src) return;
            imageData = uploadPreview.src;
        }

        // Show Loader
        resultLoader.style.display = 'flex';

        const endpoint = currentInferenceMode === 'single' ? '/predict' : '/predict_sequence';
        
        fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image: imageData,
                model_type: selectedModel
            })
        })
        .then(response => response.json())
        .then(data => {
            resultLoader.style.display = 'none';
            if (data.error) {
                console.error("API error:", data.error);
                return;
            }
            
            if (currentInferenceMode === 'single') {
                renderSingleResult(data);
            } else {
                renderSequenceResult(data);
            }
        })
        .catch(err => {
            resultLoader.style.display = 'none';
            console.error("Fetch error:", err);
        });
    }

    function renderSingleResult(data) {
        predictedValue.textContent = data.prediction;
        predictedConfidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%`;
        
        // Speak button visibility
        if (data.prediction !== 'N/A') {
            speakBtn.style.display = 'flex';
            speakBtn.onclick = () => speakText(data.prediction);
        } else {
            speakBtn.style.display = 'none';
        }

        // Preprocessed preview canvas rendering
        if (data.preprocessed_preview) {
            const previewImg = new Image();
            previewImg.onload = () => {
                pCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
                pCtx.drawImage(previewImg, 0, 0, previewCanvas.width, previewCanvas.height);
            };
            previewImg.src = 'data:image/png;base64,' + data.preprocessed_preview;
        } else {
            pCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
        }

        // Confidence Chart Bars Rendering
        if (data.distribution && data.distribution.length > 0) {
            distributionBars.innerHTML = data.distribution.map(row => `
                <div class="chart-row">
                    <div class="bar-label">${row.label}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: ${row.prob * 100}%"></div>
                    </div>
                    <div class="bar-val">${(row.prob * 100).toFixed(1)}%</div>
                </div>
            `).join('');
        } else {
            distributionBars.innerHTML = '<div class="chart-empty-state">No predictions returned.</div>';
        }
    }

    function renderSequenceResult(data) {
        sequenceValue.textContent = data.sequence || '---';
        
        if (data.sequence) {
            speakSequenceBtn.style.display = 'flex';
            speakSequenceBtn.onclick = () => speakText(data.sequence);
        } else {
            speakSequenceBtn.style.display = 'none';
        }

        // Bounding boxes annotated preview
        if (data.annotated_image) {
            annotatedSequencePreview.src = 'data:image/png;base64,' + data.annotated_image;
            annotatedSequencePreview.style.display = 'block';
            sequenceImagePlaceholder.style.display = 'none';
        } else {
            annotatedSequencePreview.style.display = 'none';
            sequenceImagePlaceholder.style.display = 'block';
        }

        // Crops decodings list
        if (data.predictions && data.predictions.length > 0) {
            segmentedListContainer.innerHTML = data.predictions.map(pred => `
                <div class="segmented-item">
                    <img class="segmented-char-img" src="data:image/png;base64,${pred.preview}" alt="crop">
                    <div class="segmented-char-val">${pred.char}</div>
                    <div class="segmented-char-conf">${(pred.confidence * 100).toFixed(0)}%</div>
                </div>
            `).join('');
        } else {
            segmentedListContainer.innerHTML = '<div class="chart-empty-state">No character segmented crops detected.</div>';
        }
    }

    // Reset UI Outputs
    function resetOutputs() {
        predictedValue.textContent = '-';
        predictedConfidence.textContent = 'Confidence: 0.0%';
        speakBtn.style.display = 'none';
        pCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
        distributionBars.innerHTML = '<div class="chart-empty-state">Draw or upload a character to view neural net activations.</div>';
        
        sequenceValue.textContent = '---';
        speakSequenceBtn.style.display = 'none';
        annotatedSequencePreview.style.display = 'none';
        sequenceImagePlaceholder.style.display = 'block';
        segmentedListContainer.innerHTML = '<div class="chart-empty-state">Individual letter crops and probabilities will be rendered here.</div>';
    }

    // Text to Speech
    function speakText(text) {
        if ('speechSynthesis' in window) {
            window.speechSynthesis.cancel(); // cancel any active speech
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.rate = 1.0;
            // For single letters, spell it clearly
            if (text.length === 1) {
                utterance.pitch = 1.1;
            }
            window.speechSynthesis.speak(utterance);
        }
    }

    // Model Status Check API call
    function checkModelStatus() {
        fetch('/api/status')
            .then(res => res.json())
            .then(data => {
                updateStatusBadge('mnist-status', 'MNIST', data.mnist.loaded);
                updateStatusBadge('emnist-status', 'EMNIST', data.emnist.loaded);
            })
            .catch(err => {
                console.error("Status check failed:", err);
                updateStatusBadge('mnist-status', 'MNIST', false, true);
                updateStatusBadge('emnist-status', 'EMNIST', false, true);
            });
    }

    function updateStatusBadge(elementId, datasetName, isLoaded, isError = false) {
        const badge = document.getElementById(elementId);
        const indicator = badge.querySelector('.status-indicator');
        const text = badge.querySelector('span:last-child');
        
        if (isError) {
            indicator.className = 'status-indicator error';
            text.textContent = `${datasetName}: Error`;
            return;
        }

        if (isLoaded) {
            indicator.className = 'status-indicator loaded';
            text.textContent = `${datasetName}: Ready`;
        } else {
            indicator.className = 'status-indicator loading';
            text.textContent = `${datasetName}: Training/Unloaded`;
        }
    }
});
