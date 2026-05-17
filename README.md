# Part 1 – Neural Network Fundamentals and Training Behavior Analysis
## Task 1: Dataset Understanding

> **Repository scope:** This repository covers **Task 1 only** of Part 1.  
> Each task/part is submitted in its own separate repository as per assignment instructions.

---

## 📂 Dataset

| Item | Detail |
|------|--------|
| **Name** | `customer_churn_nn.csv` |
| **Source** | [Google Drive – Assignment Dataset](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing) |
| **Goal** | Predict whether a customer will churn (`churn = 1`) or be retained (`churn = 0`) |
| **Modification policy** | The raw dataset is loaded **as-is**. No rows, columns, or values are altered. |

---

## 🎯 Objective

Before building any neural network we must fully understand the data.
Task 1 performs structured exploratory analysis answering six key questions:

1. How many rows and columns does the dataset have?
2. What types of input features are present?
3. What is the target variable and how is it described?
4. Are there any missing values?
5. What are the basic statistics of each feature?
6. How is the target variable distributed (visually)?

---

## 📁 Repository Structure

```
part-1-neural-network-analysis/
│
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── model_comparison_table.csv
    └── evaluation_outputs.png
```

---

## ⚙️ Setup & Usage

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd part-1-neural-network-analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place the dataset in the project root
#    Download customer_churn_nn.csv from the Google Drive link above

# 4. Open the notebook
jupyter notebook notebook.ipynb
```

Run all cells top to bottom. The final cell saves both result files into `results/`.

---

## 🔍 Approach & Steps

### Step 1 – Load the dataset
`pandas.read_csv()` loads the file with auto-inferred types. `customer_id` is flagged as a non-predictive identifier and excluded from modelling.

### Step 2 – Rows and columns
`df.shape` returns dimensions instantly. Dataset size informs batch-size choices and overfitting risk for later tasks.

### Step 3 – Feature types
Columns are classified as numerical, categorical, binary, or identifier — each group needs different preprocessing before being passed into the network.

### Step 4 – Target variable
`churn` is binary (0/1) → **binary classification** problem. Loss function will be `BCEWithLogitsLoss`; output layer will be 1 neuron with Sigmoid activation.

### Step 5 – Missing value check
`df.isnull().sum()` scanned across all 17 columns.

### Step 6 – Statistical summary
`df.describe()` extended with **Coefficient of Variation (CV% = std / |mean| × 100)** to flag features likely to cause gradient instability without standardisation.

### Step 7 – Visualisations
Five sets of plots covering: target distribution, numerical histograms, categorical breakdowns, correlation heatmap, and box plots by churn class. All combined into `results/evaluation_outputs.png`.

---

## 📊 Results & Observations

### 1. Dimensions

| Metric | Value |
|--------|-------|
| **Rows** | 2,000 |
| **Columns** | 17 (16 features + 1 target) |

---

### 2. Feature Types

| Type | Count | Columns |
|------|-------|---------|
| **Numerical** | 10 | `tenure_months`, `monthly_charges_inr`, `avg_login_days_per_month`, `support_tickets_last_90_days`, `payment_delay_days`, `data_usage_gb`, `satisfaction_score`, `last_complaint_days_ago`, `discount_percent`, `referral_count` |
| **Categorical** | 4 | `region`, `plan_type`, `contract_type`, `payment_method` |
| **Binary** | 1 | `autopay_enabled` (already 0/1, no encoding needed) |
| **Identifier** | 1 | `customer_id` — dropped before training |
| **Target** | 1 | `churn` |

---

### 3. Target Variable Description

| Property | Value |
|----------|-------|
| **Column name** | `churn` |
| **Type** | Binary integer (0 / 1) |
| **Meaning** | 0 = Customer retained · 1 = Customer churned |
| **Problem type** | Binary Classification |
| **Retained (0)** | 1,969 customers — **98.4%** |
| **Churned (1)** | 31 customers — **1.6%** |
| **Imbalance ratio** | **63.5 : 1** (majority : minority) |

> ⚠️ **Severe class imbalance.** A naive model predicting "retained" for everyone scores 98.4% accuracy but catches zero churners. Must be handled with `pos_weight` in BCELoss or SMOTE.

---

### 4. Missing Values

| Result | Detail |
|--------|--------|
| **Missing cells** | **0** — across all 17 columns |
| **Action required** | None |

✅ The dataset is fully populated. No imputation needed.

---

### 5. Basic Statistical Summary

| Feature | Mean | Std | Min | Max | CV% | Corr w/ Churn | Scale? |
|---------|------|-----|-----|-----|-----|---------------|--------|
| `tenure_months` | 25.36 | 14.13 | 1 | 72 | 55.7% | -0.106 | ⚠ Yes |
| `monthly_charges_inr` | 766.49 | 393.42 | 100 | 2157 | 51.3% | +0.062 | ⚠ Yes |
| `avg_login_days_per_month` | 18.10 | 5.40 | 1 | 30 | 29.8% | -0.021 | No |
| `support_tickets_last_90_days` | 1.95 | 1.46 | 0 | 8 | 75.0% | +0.117 | ⚠ Yes |
| `payment_delay_days` | 3.56 | 3.89 | 0 | 31 | 109.3% | -0.002 | ⚠ Yes |
| `data_usage_gb` | 90.01 | 53.22 | 0.3 | 266 | 59.1% | +0.046 | ⚠ Yes |
| `satisfaction_score` | 6.87 | 1.52 | 1 | 10 | 22.2% | -0.088 | No |
| `last_complaint_days_ago` | 46.62 | 55.07 | 0 | 424 | 118.1% | -0.005 | ⚠ Yes |
| `discount_percent` | 8.26 | 7.55 | 0 | 20 | 91.5% | +0.021 | ⚠ Yes |
| `referral_count` | 0.92 | 1.04 | 0 | 7 | 113.5% | -0.006 | ⚠ Yes |

Full table saved to → `results/model_comparison_table.csv`

**Key observations:**
- 8 of 10 features have CV% > 30 and require **z-score standardisation** before training
- `support_tickets_last_90_days` (+0.117) and `satisfaction_score` (-0.088) have the strongest correlations with churn — both intuitive signals
- `monthly_charges_inr` ranges 100–2157 INR — a 21× range that would dominate unnormalised inputs

#### Categorical Feature Breakdown

| Feature | Categories | Most Common | Note |
|---------|-----------|-------------|------|
| `region` | 5 | West (20.9%) | Fairly balanced |
| `plan_type` | 4 | Standard (35.9%) | Enterprise underrepresented (8.2%) |
| `contract_type` | 3 | Month-to-month (55.6%) | High churn-risk segment |
| `payment_method` | 5 | Credit Card (20.6%) | All roughly balanced |

---

### 6. Evaluation Outputs

![Evaluation Outputs](results/evaluation_outputs.png)

The combined plot includes:
- **A** – Target class distribution (bar chart)
- **B** – Class proportion (pie chart)
- **C** – Missing values per column (all zero ✅)
- **D–H** – Key numerical feature histograms with mean markers
- **I** – Pearson correlation of each feature with `churn`

---

## 🔑 Implications for Neural Network Design (Task 2 onwards)

| Observation | Design Decision |
|-------------|----------------|
| Severe imbalance (63.5 : 1) | `BCEWithLogitsLoss(pos_weight=tensor([63.5]))` or SMOTE |
| 4 categorical features | One-hot encode → adds ~14 input dimensions |
| 8 high-CV% numerical features | Z-score standardise with `StandardScaler` |
| Binary target (0/1) | Output: 1 neuron + Sigmoid · Loss: BCELoss |
| No missing values | No imputation layer needed |
| `customer_id` non-predictive | Drop before building input tensors |
| Metrics | Precision, Recall, F1-score, AUC-ROC (not accuracy) |

---

## 🛠️ Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scipy>=1.9.0
jupyter>=1.0.0
```

---

## 🔗 Related Repositories

| Part | Task | Repository |
|------|------|-----------|
| Part 1 | **Task 1 – Dataset Understanding** | *(this repo)* |
| Part 1 | Task 2 – Neural Network Design | `<link>` |
| Part 1 | Task 3 – Training & Backpropagation | `<link>` |
| Part 1 | Task 4 – Analysis & Visualisation | `<link>` |
