function showSelectedImage(input) {
    const selectedImage = document.getElementById('selected-image');
    selectedImage.innerHTML = '';

    if (input.files) {
        Array.from(input.files).forEach(file => {
            const reader = new FileReader();
            reader.onload = function(e) {
                const img = document.createElement('img');
                img.src = e.target.result;
                img.alt = 'Selected Image Preview';
                img.style.maxWidth = '100%';
                img.style.borderRadius = '5px';
                img.style.marginBottom = '10px';
                selectedImage.appendChild(img);
            };
            reader.readAsDataURL(file);

            const fileName = document.createElement('p');
            fileName.textContent = `Selected Image: ${file.name}`;
            selectedImage.appendChild(fileName);
        });
    }
}
