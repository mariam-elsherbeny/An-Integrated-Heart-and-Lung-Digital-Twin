async function predictPatient() {
    try {
        const data = {
            heart_rate: parseFloat(document.getElementById('heart_rate').value),
            blood_pressure: parseFloat(document.getElementById('blood_pressure').value),
            cholesterol: parseFloat(document.getElementById('cholesterol').value)
        };

        if (Object.values(data).some(value => isNaN(value))) {
            alert("❌ Please fill in all prediction fields.");
            return;
        }

        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data)
        });

        const result = await response.json();
        document.getElementById('result').innerText = `Heart Condition: ${result.condition}`;
        document.getElementById('model-viewer').setAttribute('src', result.model_url);

    } catch (error) {
        console.error("Prediction failed:", error);
        alert("❌ Prediction failed. Please try again.");
    }
}
