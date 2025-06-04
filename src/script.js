document.addEventListener('DOMContentLoaded', function() {
    // Ensure the ID here matches the ID of your <select> element in index.html
    const visualizationSelector = document.getElementById('visualizationSelector'); 
    const displayedImage = document.getElementById('displayedImage');
    const imageTitle = document.getElementById('imageTitle');
    const placeholderText = document.getElementById('placeholderText');

    // --- IMPORTANT: Define your image file paths here ---
    // The keys (e.g., 'lr_actual_vs_predicted') MUST match the 'value' attributes 
    // in your HTML <select> options.
    // The values should be the paths to your images in the 'images' folder.
    // Double-check these filenames against your actual files in the 'images' folder!
    const imagePaths = {
        // Model Performance
        'LR-actual vs predicted vehicle count': 'images/LR-actual vs predicted vehicle count.png',
        'randomForest': 'images/randomForest.png',
        'xgboost': 'images/xgboost.png',
        'RF-average vehicle count by hour of day': 'images/RF-average vehicle count by hour of day.png',
        
        // Model Comparison
        'model_comparison': 'images/compare-1.png',
        
        // Data Exploration & Time Series
        'junction1_vehicle_count_time': 'images/LR-vehicle count over junction 1.png',
        'all_junctions_volume_time': 'images/each junction volumetime.png',
        'vehicle_count_distribution': 'images/vehicle_count_distribution.png',
        
        // Add any other image key-value pairs here if you have more graphs.
        // Example:
        // 'my_other_graph_value': 'images/my_other_graph_filename.png',
    };

    visualizationSelector.addEventListener('change', function() {
        const selectedValue = visualizationSelector.value;
        // Get the human-readable text from the selected option
        const selectedOptionText = visualizationSelector.options[visualizationSelector.selectedIndex].text;

        if (selectedValue && imagePaths[selectedValue]) {
            // Check if the image path exists in our defined object
            displayedImage.src = imagePaths[selectedValue];
            displayedImage.alt = selectedOptionText; // Use the descriptive text from the option for alt text
            displayedImage.style.display = 'block'; // Make the image visible
            
            placeholderText.style.display = 'none'; // Hide the placeholder message
            
            // Update the title above the image using the descriptive text from the option
            imageTitle.textContent = selectedOptionText; 

        } else {
            // If no valid selection or image path isn't found, hide the image and show placeholder
            displayedImage.src = '#'; // Clear the src or set to a blank/default
            displayedImage.style.display = 'none'; // Hide the image element
            
            placeholderText.style.display = 'block'; // Show the placeholder message
            
            // Reset the title
            imageTitle.textContent = 'Selected Visualization'; 
        }
    });
});