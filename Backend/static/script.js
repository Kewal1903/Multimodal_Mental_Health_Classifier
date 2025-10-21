// This function runs once the HTML document is fully loaded.
document.addEventListener("DOMContentLoaded", () => {

    // --- 1. Get all our HTML elements ---
    const analyzeBtn = document.getElementById("analyze_btn");
    const loadingSpinner = document.getElementById("loading_spinner");
    const resultsSection = document.getElementById("results_section");

    const textInput = document.getElementById("text_input");
    const imageInput = document.getElementById("image_input");
    const audioInput = document.getElementById("audio_input"); // <-- ADD THIS

    const finalPrediction = document.getElementById("final_prediction");

    // Get the 'canvas' elements for our charts
    const textChartCtx = document.getElementById('text_chart').getContext('2d');
    const imageChartCtx = document.getElementById('image_chart').getContext('2d');
    const audioChartCtx = document.getElementById('audio_chart').getContext('2d'); // <-- ADD THIS

    // We'll store our chart objects here so we can update them
    let textChart, imageChart, audioChart; // <-- ADD audioChart

    // --- 2. Add "click" listener to the Analyze button ---
    analyzeBtn.addEventListener("click", async () => {

        // Get the values from the input fields
        const text = textInput.value;
        const imageFile = imageInput.files[0];
        const audioFile = audioInput.files[0]; // <-- ADD THIS

        // Basic validation
        if (!text || !imageFile || !audioFile) { // <-- ADD !audioFile
            alert("Please provide text, an image, and an audio file."); // <-- UPDATE TEXT
            return;
        }

        // Show the loading spinner and hide old results
        loadingSpinner.classList.remove("hidden");
        resultsSection.classList.add("hidden");
        analyzeBtn.disabled = true;

        // --- 3. Prepare the data to send to the API ---
        const formData = new FormData();
        formData.append("text_input", text);
        formData.append("image_input", imageFile);
        formData.append("audio_input", audioFile); // <-- ADD THIS

        try {
            // --- 4. Send the data to our FastAPI backend ---
            const response = await fetch("http://127.0.0.1:8000/predict", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }

            const results = await response.json();

            // --- 5. Update the UI with the results ---
            updateUI(results);

        } catch (error) {
            console.error("Error during analysis:", error);
            alert("An error occurred while analyzing. Please check the console.");
        } finally {
            // Hide loading spinner and re-enable button
            loadingSpinner.classList.add("hidden");
            analyzeBtn.disabled = false;
        }
    });

    // --- 6. Helper function to update all the charts and text ---
    function updateUI(results) {
        // Update the final prediction text
        finalPrediction.textContent = results.final_prediction;

        // Get the labels and data from our results
        const labels = Object.keys(results.final_probabilities);
        const textData = Object.values(results.text_probabilities);
        const imageData = Object.values(results.image_probabilities);
        const audioData = Object.values(results.audio_probabilities); // <-- ADD THIS

        // Update the charts
        textChart = createOrUpdateChart(textChart, textChartCtx, labels, textData, 'Text Probabilities');
        imageChart = createOrUpdateChart(imageChart, imageChartCtx, labels, imageData, 'Image Probabilities');
        audioChart = createOrUpdateChart(audioChart, audioChartCtx, labels, audioData, 'Audio Probabilities'); // <-- ADD THIS

        // Show the results section
        resultsSection.classList.remove("hidden");
        // Scroll to the results card
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }

    // --- 7. Helper function to create or update a bar chart ---
    function createOrUpdateChart(chartInstance, context, labels, data, title) {
        // If the chart already exists, destroy it before creating a new one
        if (chartInstance) {
            chartInstance.destroy();
        }

        // Create a new Chart.js bar chart
        return new Chart(context, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: title,
                    data: data,
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 206, 86, 0.2)',
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(153, 102, 255, 0.2)',
                        'rgba(255, 159, 64, 0.2)',
                        'rgba(199, 199, 199, 0.2)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)',
                        'rgba(255, 159, 64, 1)',
                        'rgba(199, 199, 199, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                // --- THIS IS THE FIX ---
                maintainAspectRatio: false,
                // -----------------------
                indexAxis: 'y', // Makes it a horizontal bar chart
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 1.0 // Probabilities are from 0 to 1
                    }
                },
                plugins: {
                    legend: {
                        display: false // Hides the "Text Probabilities" label
                    }
                }
            }
        });
    }
});