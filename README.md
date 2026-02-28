# 🌍 Air Quality Prediction System using LSTM

## 📌 Problem Definition
- Air pollution poses serious risks to human health.
- In urban areas, **nitrogen dioxide (NO₂)** levels significantly affect respiratory health.
- This project aims to predict NO₂ concentrations using past **temperature, humidity, carbon dioxide**, and other environmental factors.
- Approach: **Multivariate Time Series** modeling with **LSTM (Long Short-Term Memory)**.

---

## 📊 Dataset
- **Source:** [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
- **Time Span:** 1 year
- **Frequency:** Hourly measurements
- **Features:** Temperature, humidity, sensor outputs, timestamps
- **Target:** Nitrogen dioxide (NO₂) concentration

---

## ⚙️ Technologies
- **PyTorch** → LSTM modeling
- **FastAPI** → Web service (REST API)
- **Streamlit** → User interface
- **Python Libraries:** pandas, numpy, matplotlib, seaborn, scikit-learn, torch, fastapi, uvicorn, streamlit, requests

---

## 🗂️ Project Structure
```
air-quality-prediction/
│
├── load_and_explore.py     # Data loading, analysis, visualization
├── preprocessing.py        # Missing data interpolation, normalization, train/test split
├── train.py                # Model training (LSTM)
├── test.py                 # Model evaluation
├── main_api.py             # FastAPI service (predict/ endpoint)
├── test_api.py             # API tests
├── app_streamlit.py        # Streamlit user interface
├── requirements.txt        # Dependencies
└── README.md               # Documentation
```

---

## 🚀 Setup
### 1. Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
```
### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Usage

**Data Analysis**
```bash

python load_and_explore.py
```
**Preprocessing**
```bash

python preprocessing.py
```
**Train the Model**

```bash

python train.py
```

**Test the Model**

```bash

python test.py
```

**Run API Service**

```bash

uvicorn main_api:app --reload
```

- Endpoint: http://127.0.0.1:8000/predict/

**Run Streamlit UI**
```bash

streamlit run app_streamlit.py
```

## 📈 Expected Outputs

- Model Performance: Prediction accuracy for NO₂ (metrics like RMSE, MAE)

- API: JSON-based prediction results

- UI: Interactive visualizations and prediction interface

## 📝 Notes

- Missing values are handled via interpolation.

- Normalization and sliding window techniques are applied.

- Modular design ensures maintainability and scalability.