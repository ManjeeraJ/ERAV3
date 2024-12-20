let currentText = '';

async function loadFile() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file first!');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        currentText = data.text;
        document.getElementById('originalText').textContent = currentText;
        document.getElementById('preprocessBtn').disabled = false;
        document.getElementById('augmentBtn').disabled = false;
    } catch (error) {
        console.error('Error:', error);
    }
}

async function preprocessText() {
    if (!currentText) return;

    const formData = new FormData();
    formData.append('text', currentText);

    try {
        const response = await fetch('/preprocess', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        document.getElementById('preprocessedText').textContent = data.processed_text;
    } catch (error) {
        console.error('Error:', error);
    }
}

async function augmentText() {
    if (!currentText) return;

    const formData = new FormData();
    formData.append('text', currentText);

    try {
        const response = await fetch('/augment', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        document.getElementById('augmentedText').textContent = data.augmented_text;
    } catch (error) {
        console.error('Error:', error);
    }
} 