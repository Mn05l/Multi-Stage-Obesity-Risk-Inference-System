# 🏥 Multi-Stage Obesity Risk Inference System

> An end-to-end data mining pipeline that profiles health patterns through unsupervised clustering and delivers personalized risk assessments via a Fuzzy Logic Inference System optimized with a Genetic Algorithm.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Key Results](#key-results)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)
- [Sample Output](#sample-output)

---

## Overview

Obesity is one of the most pressing global public health challenges, linked to chronic diseases such as type 2 diabetes and cardiovascular disease. This project builds a multi-stage inference system that:

1. **Profiles** individuals into natural health clusters using unsupervised learning (K-Medoids & Hierarchical Clustering)
2. **Selects** the most informative features via a Genetic Algorithm
3. **Estimates** personalized health risk scores using a Fuzzy Logic Inference System

The system outputs a structured health risk report categorizing each individual as **Low**, **Moderate**, or **High** risk based on BMI, physical activity, and behavioral cluster membership.

---

## Dataset

| Property | Details |
|---|---|
| **Name** | Estimation of Obesity Levels Based on Eating Habits and Physical Condition |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/fabiosousa/obesity-levels) |
| **Records** | 2,111 individuals |
| **Features** | 17 (numeric + categorical) |
| **Target** | `NObeyesdad` — 7 obesity level classes |
| **Region** | Mexico, Peru, Colombia |
| **Missing Values** | None |

### Feature Summary

| Feature | Description | Type |
|---|---|---|
| `Age` | Age in years | Numeric |
| `Gender` | Male / Female | Categorical |
| `FAVC` | Frequent high-caloric food consumption | Categorical |
| `FCVC` | Vegetable consumption frequency (1–3) | Numeric |
| `NCP` | Number of main meals per day | Numeric |
| `CAEC` | Eating between meals | Categorical |
| `SMOKE` | Smoking habit | Categorical |
| `CH2O` | Daily water intake (1–3) | Numeric |
| `SCC` | Calorie monitoring | Categorical |
| `FAF` | Physical activity frequency (0–3) | Numeric |
| `TUE` | Daily technology use (0–2) | Numeric |
| `CALC` | Alcohol consumption frequency | Categorical |
| `MTRANS` | Main transportation mode | Categorical |
| `family_history_with_overweight` | Family history of overweight | Categorical |
| `NObeyesdad` | Obesity level — target variable | Categorical |
| `BMI` | Engineered: Weight / Height² | Numeric |

---

## Pipeline Architecture

```
Raw Data
   │
   ▼
┌─────────────────────────────┐
│   1. EDA & Visualization    │  ← Distribution, correlations, outliers
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   2. Preprocessing          │  ← BMI engineering, deduplication,
│                             │    one-hot encoding, feature fusion,
│                             │    StandardScaler
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   3. Clustering             │  ← K-Medoids (k=4) + Hierarchical
│                             │    Elbow Method + Silhouette Score
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   4. Genetic Algorithm      │  ← Feature selection via GA
│      Feature Selection      │    Fitness = Silhouette Score
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   5. Fuzzy Logic Inference  │  ← Inputs: BMI, FAF, Cluster
│                             │    Output: Health Risk Score (0–100)
└────────────┬────────────────┘
             │
             ▼
    Risk Report (JSON)
    Low / Moderate / High
```

---

## Project Structure

```
obesity-risk-inference/
│
├── notebook/
│   └── obesity_risk_pipeline.ipynb     ← Main project notebook
│
├── data/
│   └── ObesityDataSet_raw_and_data_sinthetic.csv
│
└── README.md
```

---

## Key Results

### EDA Findings
- Dataset is **well-balanced** across all 7 obesity classes (272–351 records each)
- **No missing values** across all 2,111 records
- **81.8%** of individuals have a family history of overweight — the strongest single predictor
- Physical activity (FAF) drops consistently with obesity severity: Normal Weight averages **~1.5 days/week** vs. Obesity Type III at **~0.7 days/week**
- Outliers are minimal: fewer than **3% per feature**, treated with Winsorization

### Clustering (K-Medoids, k=4)

| Cluster | Clinical Profile | Avg BMI | Avg FAF | Avg Age |
|---|---|---|---|---|
| 0 | Obese Class III | ~42.1 | ~0.66 | ~22 |
| 1 | Obese Class I | ~32.7 | ~0.85 | ~31.7 |
| 2 | Overweight | ~25.5 | ~1.25 | ~21.7 |
| 3 | Normal Weight (At-Risk) | ~24.9 | ~0.75 | ~21 |

### Genetic Algorithm (Feature Selection)

| Setting | Value |
|---|---|
| Population size | 30 chromosomes |
| Generations | 20 |
| Mutation rate | 20% |
| Fitness metric | Silhouette Score (KMedoids, k=4) |

### Fuzzy Logic System

| Input | Fuzzy Sets |
|---|---|
| BMI | underweight / normal / overweight / obese |
| FAF | low / medium / high |
| Cluster | c0 / c1 / c2 / c3 |
| **Output: Health Risk** | **low / medium / high (0–100 scale)** |

Risk thresholds: `< 35` → Low Risk &nbsp;|&nbsp; `35–65` → Moderate Risk &nbsp;|&nbsp; `> 65` → High Risk

---

## Technologies Used

| Category | Libraries |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Clustering | `scikit-learn-extra` (KMedoids), `scipy` (Hierarchical) |
| Fuzzy Logic | `scikit-fuzzy` |
| Feature Selection | Custom Genetic Algorithm (`numpy`, `random`) |
| Preprocessing | `sklearn.preprocessing` (StandardScaler, LabelEncoder) |

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/obesity-risk-inference.git
cd obesity-risk-inference
```

### 2. Install dependencies
```bash
pip install numpy==1.26.4 pandas matplotlib seaborn scikit-learn scipy scikit-fuzzy scikit-learn-extra
```

> ⚠️ `numpy==1.26.4` is pinned intentionally — `scikit-learn-extra` requires this specific version to avoid import conflicts.

### 3. Add the dataset
Place `ObesityDataSet_raw_and_data_sinthetic.csv` in the `data/` folder.  
Download from [Kaggle](https://www.kaggle.com/datasets/fabiosousa/obesity-levels).

### 4. Run the notebook
```bash
jupyter notebook notebook/obesity_risk_pipeline.ipynb
```

Run all cells in order. The notebook is self-contained and fully documented.

---

## Sample Output

```json
{
    "BMI": 27.4,
    "Physical Activity (FAF)": 0.8,
    "Predicted Cluster": 1,
    "Fuzzy Risk Score": 71.34,
    "Final Assessment: you have ": "High Risk"
}
```

---

## Authors
---
[@Manahil Elhadi](https://github.com/mn05l)
Alexandria University, Faculty of Computers and Data Science  
Data Mining Course Project — 2025/2026
