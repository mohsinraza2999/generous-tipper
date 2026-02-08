# Generous Tip Giver Prediction

## Problem
Taxi ride-hailing platforms rely heavily on tips as a key component of driver income, yet passenger tipping behavior is highly variable and difficult to predict. This unpredictability limits the platform’s ability to optimize driver–rider matching, incentives, and service quality. Large volumes of trip, fare, temporal, and behavioral data are generated but remain underutilized for tipping prediction. A data science and machine learning approach can identify patterns that distinguish generous tippers from others. Ultimately, this leads to higher service quality, better retention, and increased platform efficiency.

## Solution
Built a full ML pipeline including:
- Data ingestion & cleaning
- Feature engineering
- Model training (XGBoost, Random Forest, Logistic Regression)
- Fast API deployment
- Dockerized application

## 📊 Dataset

* **Type:** Yellow Taxi Trip dataset from kaggle
* **Target:** Generous Tipper
* **Features:** Eighteen Numerical and encoded categorical attributes
* **Size:** 22700 Observations

## Tech Stack
Python, Pandas, Scikit-learn, XGBoost, FastAPI, Docker

## Architecture
```text
generous-tipper/
│
├── data/                 # raw & processed data
├── config/               # data & training configurations
├── frontend/             # Core frontend logic with dockerization
├── notebooks/            # Training and data cleaning notebooks
├── src/                  # Core data, training and backend pipeline logic
├── tests/                # Basic unit tests of data, training, api pipelines
├── docker-compose.yaml   # dockerizing back and frontend with health check every 30 seconds
├── Dockerfile            # multi-step dockerization for clean containerization
├── pyproject.toml
├── README.md
└── LICENSE
```

---

## 🚀 Quick Start

```bash
git clone https://github.com/mohsinraza2999/generous-tipper.git
cd house-price-prediction
python src/cli.py preprocess
python src/cli.py train
python src/cli.py route
```

---

## 🔮 Making Predictions
```bash
python src/cli.py route
```
For only backend and Swagger UI.
```text
http://localhost:8000/docs
```
Example response:

```json
{
  "prediction": "generous",
  "processed_at": "10-02-2026T07:30:21S",
  "latency_ms": 0.04
}
```

---

## 🧪 Testing

Run all unit and integration tests:

```bash
pip install pytest
pytest tests/
```

Tests cover:

* Data preprocessing pipeline
* API routes
* Model inference behavior

---

## 🧱 Docker Build
Dockerize back and frontend. Also check health in every 30 seconds.
```bash
docker-compose up --build
```

1. Run in browser for both front and backend
```text
http://localhost:3000 
```
2. For only backend and Swagger UI.
```text
http://localhost:8000/docs
```
Example response:

```json
{
  "prediction": "generous",
  "processed_at": "10-02-2026T07:30:21S",
  "latency_ms": 0.04
}
```


---

## 🔧 Configuration

* All hyperparameters stored in YAML files
* Data paths, training parameters, and inference behavior configurable
* Environment-agnostic (local or containerized)

---

## 🧠 Design Decisions & Trade-offs

* **Why Dachine Learning?**
  Beause tree-based models perform well on tabular data, so neural networks are not chosen to practice model abstraction, extensibility, and deployment workflows.

* **Why config-driven pipelines?**
  To separate experimentation from code changes and improve reproducibility.

* **Why both CLI and scripts?**
  CLI serves developers; scripts support automation and CI.

---

## Future Improvements
  * Model monitoring & drift detection
  * Cloud deployment

---

## 🧠 Key Learnings

* ML systems should be designed as maintainable software
* Testing pipelines prevents silent failures
* Separation of training and inference is critical

---

## 📜 CI & Automation

* GitHub Actions pipeline:
  * Runs tests on push
  * Ensures build stability
* Docker build validation included

---

## 📬 Contact

**Author:** Mohsin Raza
**Target Role:** Machine Learning Engineer / AI Engineer
**GitHub:** [github/mohsinraza2999](https://github.com/mohsinraza2999)
**LinkedIn:** *[linkedin/mohsin-raza](https://www.linkedin.com/in/mohsin-raza-b7ab73328)*