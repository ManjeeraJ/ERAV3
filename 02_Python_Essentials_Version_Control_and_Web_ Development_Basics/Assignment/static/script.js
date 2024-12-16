document.querySelectorAll('input[name="animal"]').forEach(radio => {
    radio.addEventListener('change', function() {
        const imageContainer = document.getElementById('image-container');
        const selectedAnimal = this.value;
        
        imageContainer.innerHTML = `<img src="/static/images/${selectedAnimal}.jpg" alt="${selectedAnimal}">`;
    });
});

async function uploadFile() {
    const fileInput = document.getElementById('file-input');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file first');
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
        document.getElementById('file-info').innerHTML = `
            <p>File Name: ${data.filename}</p>
            <p>File Size: ${data.size} bytes</p>
            <p>File Type: ${data.content_type}</p>
        `;
    } catch (error) {
        console.error('Error:', error);
        alert('Error uploading file');
    }
} 