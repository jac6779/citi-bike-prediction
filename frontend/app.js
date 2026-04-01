const form = document.getElementById("predict-form");
const resultBox = document.getElementById("result");
const predictionText = document.getElementById("prediction-text");
const rawResponse = document.getElementById("raw-response");

const API_URL = "/predict";

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const payload = {
    agency: document.getElementById("agency").value,
    borough: document.getElementById("borough").value,
    complaint_day: Number(document.getElementById("complaint_day").value),
    complaint_hr: Number(document.getElementById("complaint_hr").value),
    complaint_month: Number(document.getElementById("complaint_month").value),
    complaint_type: document.getElementById("complaint_type").value.trim(),
    descriptor: document.getElementById("descriptor").value.trim(),
    incident_zip: document.getElementById("incident_zip").value.trim(),
    latitude: Number(document.getElementById("latitude").value),
    location_type: document.getElementById("location_type").value.trim(),
    longitude: Number(document.getElementById("longitude").value)
  };

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(payload)
    });

    const text = await response.text();

    let data;
    try {
      data = JSON.parse(text);
    } catch {
      data = text;
    }

    resultBox.classList.remove("hidden");

    if (!response.ok) {
      predictionText.textContent = `Request failed with status ${response.status}`;
      rawResponse.textContent =
        typeof data === "string" ? data : JSON.stringify(data, null, 2);
      return;
    }

    const prediction =
      data.predicted_resolution_time ??
      data.prediction ??
      data.predicted_value ??
      data.result;

    predictionText.textContent = prediction
      ? `Prediction: ${prediction}`
      : "Prediction received.";

    rawResponse.textContent = JSON.stringify(data, null, 2);
  } catch (error) {
    resultBox.classList.remove("hidden");
    predictionText.textContent = "Network/browser request failed.";
    rawResponse.textContent = String(error);
  }
});