# 🏠 House Price Prediction — End-to-End Data Analytics Project

> **An interactive machine-learning dashboard for analysing and predicting residential house prices.**

---

## 📌 Table of Contents

1. [Executive Summary](#-executive-summary)
2. [Problem Statement](#-problem-statement)
3. [Dataset](#-dataset)
4. [Project Structure](#-project-structure)
5. [Tech Stack](#-tech-stack)
6. [Setup Guide](#-setup-guide)
7. [Dashboard Pages](#-dashboard-pages)
8. [ML Models & Metrics](#-ml-models--metrics)
9. [Key Findings](#-key-findings)
10. [License](#-license)

---

## 📝 Executive Summary

Housing markets generate vast quantities of structured data—property size, age, location, and condition—that lend themselves well to machine-learning-based pricing models. This project delivers a **production-ready, interactive Streamlit dashboard** that walks a user through the complete data science life-cycle:

| Phase | Output |
|---|---|
| Data Ingestion & Cleaning | Cleaned DataFrame with derived features |
| Exploratory Data Analysis | Correlation heatmaps, distribution plots, comparative charts |
| Model Training & Evaluation | Linear Regression vs Random Forest with R², MAE, RMSE |
| Live Prediction | Interactive form with gauge chart and model comparison |

The **Random Forest Regressor** achieves superior predictive performance and is used as the primary estimator in the prediction interface.

---

## ❓ Problem Statement

Accurately estimating residential property prices is critical for:

- **Buyers** — avoid over-paying in competitive markets.
- **Sellers** — set data-driven asking prices.
- **Investors** — identify undervalued properties.
- **Lenders** — underwrite mortgage risk.

Traditional valuation methods (comparable sales, manual appraisals) are slow and subjective. This project builds a **supervised ML regression pipeline** trained on structured property attributes to predict prices quickly and objectively.

**Goal:** Given a set of house features (area, bedrooms, location, condition, etc.), predict the market price in USD.

---

## 📂 Dataset

| Property | Value |
|---|---|
| **Source** | Kaggle |
| **Link** | [House Price Prediction Dataset](https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset) |
| **File** | `archive/House Price Prediction Dataset.csv` |
| **Rows** | ~1 000 (cleaned) |
| **Columns** | 10 raw → 13 after feature engineering |

### Raw Columns

| Column | Type | Description |
|---|---|---|
| `Id` | int | Row identifier (dropped) |
| `Area` | int | Living area in square feet |
| `Bedrooms` | int | Number of bedrooms |
| `Bathrooms` | int | Number of bathrooms |
| `Floors` | int | Number of floors |
| `YearBuilt` | int | Year the house was built |
| `Location` | str | Neighbourhood type: Downtown, Suburban, Urban, Rural |
| `Condition` | str | House condition: Excellent, Good, Fair, Poor |
| `Garage` | str | Garage availability: Yes / No |
| `Price` | int | Target — sale price in USD |

### Engineered Features

| Feature | Formula |
|---|---|
| `HouseAge` | `2024 − YearBuilt` |
| `PricePerSqFt` | `Price ÷ Area` |
| `RoomsTotal` | `Bedrooms + Bathrooms` |

---

## 📁 Project Structure

```
Data_Analytics_Project/
├── archive/
│   └── House Price Prediction Dataset.csv   # Raw dataset
├── app.py                                   # Streamlit dashboard (all-in-one)
├── requirements.txt                         # Python dependencies
├── README.md                                # This file
└── project_report.docx                      # Detailed project report
```

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | Interactive web dashboard |
| `pandas` | Data manipulation |
| `numpy` | Numerical operations |
| `scikit-learn` | ML models (LinearRegression, RandomForestRegressor) |
| `plotly` | Interactive charts |
| `statsmodels` | OLS trendlines in scatter plots |
| `openpyxl` | Excel export support |

---

## 🚀 Setup Guide

### Prerequisites

- Python **3.10 or later**
- `pip` package manager

### 1. Clone / Download the project

```bash
git clone <your-repo-url>
cd Data_Analytics_Project
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the dataset

Ensure the CSV file is located at:
```
archive/House Price Prediction Dataset.csv
```

Download it from: [https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset](https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset)

### 5. Run the dashboard

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## 📊 Dashboard Pages

| Page | Description |
|---|---|
| **🏠 Home** | KPI cards, project pipeline overview, price distribution chart |
| **📋 Data Overview** | Raw data table, descriptive statistics, data-quality report, categorical distributions |
| **📊 EDA** | Correlation heatmap, scatter plots, box/violin charts, feature–price relationships |
| **🤖 Model Performance** | Side-by-side R²/MAE/RMSE comparison, Actual vs Predicted scatter, residual histogram, feature importance |
| **🔮 Predict Price** | Interactive form → Random Forest prediction with gauge chart and model comparison |

---

## 🤖 ML Models & Metrics

Two regression models are trained and compared on an 80 / 20 train–test split (random seed 42).

### Linear Regression

- Baseline model
- Assumes linear relationship between features and price

### Random Forest Regressor

- Ensemble of 200 decision trees
- Captures non-linear interactions
- Provides feature importance rankings
- Used as the primary prediction model

### Evaluation Metrics

| Metric | Formula | Goal |
|---|---|---|
| **R²** | 1 − SS_res / SS_tot | Closer to 1 |
| **MAE** | mean\|actual − predicted\| | Lower |
| **RMSE** | √mean(actual − predicted)² | Lower |

---

## 🔍 Key Findings

1. **Area** is the single most important predictor of price according to feature importance.
2. **House Age** shows a moderate negative correlation with price — newer homes tend to be pricier.
3. **Condition** significantly stratifies price: *Excellent* condition homes command a clear premium.
4. **Location** affects price spread rather than the median — Downtown has the widest range.
5. **Garage** availability adds a modest price premium on average.
6. Random Forest consistently outperforms Linear Regression on both R² and error metrics.

---

## 📄 License

This project is released under the [MIT License](https://opensource.org/licenses/MIT).
Dataset is sourced from Kaggle and subject to its original terms of use.
