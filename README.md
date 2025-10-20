# 📘 Predicting Customer Churn in Telecommunications

## 🧠 Overview

This project applies data analytics and machine learning to predict
customer churn in a telecommunications company. By identifying customers
likely to leave, the business can proactively target retention efforts
and improve long-term profitability.

------------------------------------------------------------------------

## 🎯 Objectives

-   Analyze customer behavior and churn patterns.
-   Build a predictive model to classify customers at risk of leaving.
-   Provide actionable insights through Power BI and Python dashboards.
-   Help marketing and retention teams reduce churn rate.

------------------------------------------------------------------------

## 📊 Dataset

**Source:** [Telco Customer Churn Dataset
(Kaggle)](https://www.kaggle.com/blastchar/telco-customer-churn)

  Feature           Description
  ----------------- ----------------------------------------------------
  customerID        Unique ID for each customer
  gender            Male or Female
  SeniorCitizen     Whether the customer is a senior citizen
  Partner           Whether the customer has a partner
  Dependents        Whether the customer has dependents
  tenure            Number of months the customer has stayed
  PhoneService      Whether the customer has phone service
  MultipleLines     Whether the customer has multiple lines
  InternetService   Type of internet service
  OnlineSecurity    Whether online security is included
  TechSupport       Whether tech support is included
  Contract          Contract type (Month-to-month, One year, Two year)
  PaymentMethod     Payment method used by the customer
  MonthlyCharges    Monthly amount charged
  TotalCharges      Total amount charged
  Churn             Target variable (Yes/No)

------------------------------------------------------------------------

## ⚙️ Project Structure

    Predicting-Customer-Churn/
    │
    ├── data/
    │   └── telco_churn.csv
    │
    ├── notebooks/
    │   └── exploratory_analysis.ipynb
    │
    ├── scripts/
    │   ├── load_data.py
    │   ├── clean_data.py
    │   ├── feature_engineering.py
    │   ├── train_model.py
    │   ├── evaluate_model.py
    │   └── app.py
    │
    ├── models/
    │   └── churn_model.pkl
    │
    ├── requirements.txt
    ├── README.md
    └── PowerBI_Dashboard.pbix

------------------------------------------------------------------------

## 🧩 Workflow

### 1. Data Collection

``` python
df = pd.read_csv('data/telco_churn.csv')
```

### 2. Data Cleaning & Transformation

``` python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()
```

### 3. Feature Engineering

-   Encode categorical variables.
-   Create tenure buckets.
-   Standardize numerical features.

### 4. Model Training

``` python
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
```

### 5. Model Evaluation

``` python
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
```

### 6. Deployment

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 📈 Power BI Dashboard

The Power BI dashboard includes: - **Customer Churn Overview:** Overall
churn rate and customer segments. - **Demographic Analysis:** Churn by
gender, age, contract type. - **Revenue Impact:** Lost revenue due to
churn. - **Predictive Model Output:** Churn probability distribution.

------------------------------------------------------------------------

## 🚀 Reproducibility

**Setup environment:**

``` bash
git clone https://github.com/yourusername/predict-customer-churn.git
cd predict-customer-churn
pip install -r requirements.txt
```

**Run training:**

``` bash
python scripts/train_model.py
```

**Run app:**

``` bash
streamlit run scripts/app.py
```

------------------------------------------------------------------------

## 📦 Requirements

    pandas
    numpy
    scikit-learn
    matplotlib
    seaborn
    streamlit

------------------------------------------------------------------------

## 📜 License

MIT License

------------------------------------------------------------------------

## 👨‍💻 Author

**Bahre Hailemariam**\
*Data Analyst \| BI Developer*\
📧 your.email@example.com\
🌐 your-portfolio-link.com
