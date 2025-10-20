# 📘 Predicting Customer Churn in Telecommunications

## 🧠 Project Overview
Customer churn — when subscribers discontinue their services — is one of the most critical challenges for telecom companies.  
This project applies **data analytics**, **machine learning**, and **business intelligence (BI)** to predict churn probability, uncover the key drivers behind it, and support **retention-focused decision-making**.

By predicting which customers are at risk, telecom operators can reduce churn, increase customer lifetime value (CLV), and optimize marketing campaigns.

---

## 🎯 Business Objectives
1. Identify customers likely to leave the service.
2. Understand which features and behaviors influence churn (e.g., tenure, contract type, payment method).
3. Design data-driven retention campaigns targeting high-risk customers.
4. Build predictive models to automate churn detection.
5. Deliver insights through a **Power BI dashboard** and **Python-based reports** for ongoing monitoring.

---

## 🧩 Data Source
**Dataset:** [Telco Customer Churn (Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)

This dataset contains demographic, service usage, and billing details for telecom customers.

| Feature | Description | Example |
|----------|-------------|----------|
| `customerID` | Unique customer identifier | 7590-VHVEG |
| `gender` | Male or Female | Male |
| `SeniorCitizen` | 1 if senior, 0 otherwise | 0 |
| `Partner` | Yes/No | Yes |
| `Dependents` | Yes/No | No |
| `tenure` | Number of months as customer | 34 |
| `PhoneService` | Yes/No | Yes |
| `MultipleLines` | Yes/No/No phone service | No |
| `InternetService` | DSL/Fiber optic/None | Fiber optic |
| `OnlineSecurity` | Yes/No/No internet service | No |
| `TechSupport` | Yes/No/No internet service | No |
| `Contract` | Month-to-month/One year/Two year | Month-to-month |
| `PaymentMethod` | Payment method type | Electronic check |
| `MonthlyCharges` | Monthly bill amount | 70.35 |
| `TotalCharges` | Total amount billed | 1397.47 |
| `Churn` | Target variable (Yes/No) | Yes |

---

## ⚙️ Tech Stack
| Category | Tools / Libraries |
|-----------|------------------|
| **Data Processing** | Pandas, NumPy, Power Query |
| **Exploratory Analysis** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost, RandomForest |
| **Model Deployment** | Streamlit |
| **Dashboarding** | Power BI |
| **Version Control** | Git, GitHub |

---

## 🧠 Data Workflow
1️⃣ **Data Extraction & Loading**  

```bash
import pandas as pd
df = pd.read_csv('data/telco_churn.csv')
```
2️⃣ **Data Cleaning**  
```bash
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)
```
3️⃣ **Feature Engineering**  
```bash

from sklearn.preprocessing import OneHotEncoder, StandardScaler

```
4️⃣ **Model Development** 
```bash
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

```
5️⃣ **Model Evaluation** 
```bash
from sklearn.metrics import classification_report, roc_auc_score
print(classification_report(y_test, y_pred))
print('AUC:', roc_auc_score(y_test, y_proba))

```

---

## 📈 Power BI Dashboard
Power BI visualizes actionable insights:
- Customer Segmentation
- Financial Impact
- Predictive Analysis
- Geo Analysis

---

## 🚀 Deployment
### Local Streamlit App:
```bash
streamlit run scripts/app.py
```
### Power BI Integration:
1. Export churn predictions from Python.
2. Load into Power BI as a data source.
3. Build visuals using DAX & Power Query.

---

## 🔁 Reproducibility
### Setup:
```bash
git clone https://github.com/yourusername/predict-customer-churn.git
cd predict-customer-churn
pip install -r requirements.txt
```
### Train Model:
```bash
python scripts/train_model.py
```
### Launch Dashboard:
```bash
streamlit run scripts/app.py
```

---

## 📊 Model Insights
| Feature | Importance |
|----------|-------------|
| Contract Type | 0.28 |
| Tenure | 0.23 |
| MonthlyCharges | 0.17 |
| Payment Method | 0.12 |
| Internet Service | 0.10 |
| Tech Support | 0.07 |

---

## 🧠 Business Recommendations
- Offer **discounted annual contracts**.
- Improve customer service for fiber users.
- Target **senior citizens** with simplified billing.
- Launch **loyalty rewards programs**.

---

## 📜 License
This project is licensed under the **MIT License**.

---

## 👨‍💻 Author
**Bahre Hailemariam**  
_Data Analyst | BI Developer_  
📧 your.email@example.com  
🌐 [your-portfolio-link.com](#)  
💼 [LinkedIn Profile](#)
