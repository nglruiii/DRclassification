document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const previewContainer = document.getElementById('preview-container');
    const imagePreview = document.getElementById('image-preview');
    const removeBtn = document.getElementById('remove-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    const resultSection = document.getElementById('result-section');
    const loader = document.getElementById('loader');
    const resultContent = document.getElementById('result-content');
    const severityBadge = document.getElementById('severity-badge');
    const severityLabel = document.getElementById('severity-label');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceFill = document.getElementById('confidence-fill');

    let currentFile = null;

    // Severity color mapping
    const colors = {
        0: 'var(--sev-0)',
        1: 'var(--sev-1)',
        2: 'var(--sev-2)',
        3: 'var(--sev-3)',
        4: 'var(--sev-4)'
    };

    // Drag and Drop Events
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => {
            dropZone.classList.remove('dragover');
        }, false);
    });

    dropZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        handleFiles(files);
    });

    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    fileInput.addEventListener('change', function() {
        handleFiles(this.files);
    });

    removeBtn.addEventListener('click', resetUI);

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        currentFile = file;
        
        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            dropZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');
            resultSection.classList.add('hidden');
        };
        reader.readAsDataURL(file);
    }

    function resetUI() {
        currentFile = null;
        fileInput.value = '';
        dropZone.classList.remove('hidden');
        previewContainer.classList.add('hidden');
        resultSection.classList.add('hidden');
    }

    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI State: Loading
        resultSection.classList.remove('hidden');
        loader.classList.remove('hidden');
        resultContent.classList.add('hidden');
        confidenceFill.style.width = '0%';
        analyzeBtn.disabled = true;

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.detail || `Server error: ${response.statusText}`);
            }

            const data = await response.json();
            displayResult(data);
        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during analysis: ' + error.message);
            resultSection.classList.add('hidden');
        } finally {
            analyzeBtn.disabled = false;
        }
    });

    function displayResult(data) {
        // UI State: Success
        loader.classList.add('hidden');
        resultContent.classList.remove('hidden');

        // Update values
        severityLabel.textContent = data.label;
        
        const confPercent = (data.confidence * 100).toFixed(1);
        confidenceValue.textContent = `${confPercent}%`;
        
        // Apply color
        const color = colors[data.class_index] || 'var(--primary-color)';
        severityBadge.style.backgroundColor = color;
        severityBadge.style.boxShadow = `0 0 20px ${color}80`;
        severityBadge.style.color = data.class_index === 1 ? '#000' : '#fff'; // Adjust text color for yellow
        confidenceFill.style.backgroundColor = color;

        // Animate progress bar
        setTimeout(() => {
            confidenceFill.style.width = `${confPercent}%`;
        }, 100);
    }
});
