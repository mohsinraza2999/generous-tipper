const form = document.getElementById("prediction-form");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");
const loadingBox = document.getElementById("loading");

const API_URL = "http://backend:8000/predict";

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  resultBox.classList.add("hidden");
  errorBox.classList.add("hidden");
  loadingBox.classList.remove("hidden");

  const formData = new FormData(form);
  const payload = {};

  formData.forEach((value, key) => {
    if (key === "trip_date") {
      payload[key] = new Date(value).toISOString();
    } else {
      payload[key] = Number(value);
    }
  });

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      throw new Error(await res.text());
    }

    const data = await res.json();

    document.getElementById("prediction").textContent = data.prediction;
    document.getElementById("latency").textContent = data.latency_ms.toFixed(2);
    document.getElementById("processed").textContent =
      new Date(data.processed_at).toLocaleString();

    resultBox.classList.remove("hidden");
  } catch (err) {
    errorBox.textContent = err.message || "Prediction failed";
    errorBox.classList.remove("hidden");
  } finally {
    loadingBox.classList.add("hidden");
  }
});