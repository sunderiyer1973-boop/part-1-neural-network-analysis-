# Part 1 – Neural Network Fundamentals and Training Behavior Analysis

**Dataset:** [Customer Churn Neural Network Dataset](https://drive.google.com/drive/folders/1akV6po4Nrgkc3yQrJkzA6cJlV-wBvUYs?usp=sharing)  
**Goal:** Predict whether a customer will churn (`churn = 1`) or be retained (`churn = 0`)  
**Problem Type:** Binary Classification  
**Library:** scikit-learn `MLPClassifier` (feed-forward neural network)

> The raw dataset is loaded as-is. No rows, columns, or values are modified.

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
git clone <your-repo-url>
cd part-1-neural-network-analysis
pip install -r requirements.txt
# Place customer_churn_nn.csv in the project root
jupyter notebook notebook.ipynb
```

Run all cells top to bottom. Results are saved automatically to `results/`.

---

## Task 1: Dataset Understanding

### Approach
Load the raw CSV and explore its structure, types, statistics, and target distribution before any modelling.

### Steps
1. Load with `pd.read_csv()` — no modifications
2. Inspect shape, column names, and inferred dtypes
3. Classify columns: numerical, categorical, binary, identifier, target
4. Describe target variable and identify problem type
5. Check missing values with `df.isnull().sum()`
6. Compute statistical summary extended with CV%
7. Visualise: target distribution, feature histograms, correlation heatmap

### Results

| Metric | Value |
|--------|-------|
| Rows | 2,000 |
| Columns | 17 (16 features + 1 target) |
| Numerical features | 10 |
| Categorical features | 4 |
| Binary features | 1 (`autopay_enabled`) |
| Identifier (dropped) | `customer_id` |
| Missing values | **0** — no imputation needed |

**Target variable (`churn`):**
- Retained (0): 1,969 — 98.4%
- Churned (1): 31 — 1.6%
- Imbalance ratio: **63.5 : 1** — severe imbalance

### Observations
- `support_tickets_last_90_days` (r = +0.117) and `satisfaction_score` (r = −0.088) show the strongest correlations with churn
- 8 of 10 numerical features have CV% > 30, making standardisation essential
- Over half the customers are on Month-to-month contracts — a known churn-risk segment
- Standard accuracy is a misleading metric here; AUC-ROC, Recall, and F1-score must be used

---

## Task 2: Data Preprocessing

### Approach
Transform the raw dataset into a form the neural network can consume, without leaking test information into training.

### Steps & Decisions

| Step | Action | Reason |
|------|--------|--------|
| Missing values | None found — assertion passes | No imputation required |
| Drop identifier | Remove `customer_id` | No predictive value; would cause ID memorisation |
| One-hot encode | `pd.get_dummies(drop_first=True)` on 4 categorical columns | Convert strings to binary; `drop_first` avoids dummy variable trap |
| Train/test split | 80/20 stratified (`stratify=y`, `random_state=42`) | Done before scaling to prevent leakage; stratification preserves 1.6% churn rate |
| Z-score scaling | `StandardScaler` fit on train only, applied to both | Prevents gradient dominance by large-range features |
| Sample weights | `compute_sample_weight('balanced')` | Compensates for 63:1 imbalance in MLPClassifier |

### Results

| Object | Shape | Description |
|--------|-------|-------------|
| `X_train_scaled` | (1600, 24) | Training features |
| `X_test_scaled` | (400, 24) | Test features |
| `y_train` | (1600,) | 1.6% churn rate preserved |
| `y_test` | (400,) | 1.5% churn rate preserved |

24 input features = 10 numerical + 1 binary + 13 one-hot encoded dummies

---

## Task 3: Neural Network Model Building

### Architecture

```
Input Layer    →  24 neurons   (one per feature after encoding)
Hidden Layer 1 →  64 neurons   + ReLU activation
Hidden Layer 2 →  32 neurons   + ReLU activation
Output Layer   →   1 neuron    + Sigmoid activation
```

### Design Decisions

| Component | Choice | Reason |
|-----------|--------|--------|
| **Activation (hidden)** | ReLU | Avoids vanishing gradient; fast convergence; industry standard |
| **Activation (output)** | Sigmoid | Squashes output to [0,1]; interpretable as churn probability |
| **Loss function** | Binary Cross-Entropy | Standard for binary classification problems |
| **Optimizer** | Adam | Adaptive per-parameter learning rates; handles sparse gradients well |
| **Early stopping** | Enabled (patience=10) | Prevents overfitting; stops when validation score plateaus |
| **Class imbalance** | Sample weights (balanced) | Upweights the 31 churn examples 63× relative to retained |

### Implementation
```python
MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=300,
    batch_size=32,
    early_stopping=True,
    validation_fraction=0.1,
    random_state=42
)
```

---

## Task 4: Training and Evaluation

### Baseline Model Results

| Metric | Value |
|--------|-------|
| Train Accuracy | 0.9831 |
| Test Accuracy | 0.9850 |
| Precision | 0.0000 |
| Recall | 0.0000 |
| F1-Score | 0.0000 |
| **AUC-ROC** | **0.7458** |
| Epochs run | 12 (early stopped) |

### Confusion Matrix (Baseline)

```
              Predicted
              Retained   Churned
Actual  Retained   394        0
        Churned      6        0
```

### Interpretation
- The baseline model collapses to predicting "Retained" for all samples — the classic failure mode with severe class imbalance
- Despite sample weighting, the default LR (0.001) with ReLU is too conservative — it minimises loss by never predicting the rare class
- AUC-ROC = 0.746 shows the model *does* learn some signal in its probability scores, but the decision threshold (0.5) is far too high for a 1.6% prevalence class
- **Accuracy is misleading** — 98.5% accuracy = 0% churn detection rate

---

## Task 5: Hyperparameter Experimentation

### Experiment Configurations

| # | Experiment | Architecture | Activation | LR | Batch | What Changes |
|---|-----------|-------------|-----------|-----|-------|-------------|
| 1 | Baseline | (64, 32) | ReLU | 0.001 | 32 | Reference |
| 2 | Deeper Network | (128, 64, 32) | ReLU | 0.001 | 32 | More layers/neurons |
| 3 | Higher LR | (64, 32) | ReLU | 0.01 | 32 | 10× higher LR |
| 4 | Lower LR | (64, 32) | ReLU | 0.0001 | 32 | 10× lower LR |
| 5 | tanh Activation | (64, 32) | tanh | 0.001 | 32 | Activation function |
| 6 | Large Batch | (64, 32) | ReLU | 0.001 | 128 | 4× larger batch |

### Comparison Table

| Experiment | AUC-ROC | Recall | F1-Score | Precision | Test Acc |
|-----------|---------|--------|----------|-----------|----------|
| Exp 1: Baseline | 0.7458 | 0.0000 | 0.0000 | 0.0000 | 0.9850 |
| Exp 2: Deeper Network | 0.8752 | 0.1667 | 0.2000 | 0.2500 | 0.9800 |
| Exp 3: Higher LR | 0.9095 | 0.8333 | 0.1266 | 0.0685 | 0.8275 |
| Exp 4: Lower LR | 0.7246 | 0.5000 | 0.0659 | 0.0353 | 0.7875 |
| **Exp 5: tanh** | **0.9298** | **0.8333** | **0.1852** | **0.1042** | **0.8900** |
| Exp 6: Large Batch | 0.8494 | 0.5000 | 0.1071 | 0.0600 | 0.8750 |

### Observations

- **Exp 1 (Baseline):** Collapses to predicting all "retained". Zero recall despite sample weights
- **Exp 2 (Deeper):** Extra layer adds capacity — AUC jumps to 0.875, first model to detect any churner
- **Exp 3 (Higher LR):** Best recall (0.833) — higher LR forces the model to commit to minority class predictions. Low precision (many false positives)
- **Exp 4 (Lower LR):** Worst AUC (0.725) — too slow to learn; early stopping fires before convergence
- **Exp 5 (tanh) — WINNER:** Best AUC (0.930) and best F1 (0.185). tanh's zero-centered output [-1,1] handles the rare churn class more effectively than ReLU at this dataset size
- **Exp 6 (Large Batch):** Smoothed gradients reduce sensitivity to the minority class; moderate performance

**Best model: Exp 5 — tanh, LR=0.001, Architecture (64,32)**

---

## Task 6: Final Reflection

### 6.1 What role do weights and biases play?

**Weights** are the learnable parameters on each connection. Each neuron computes:
`z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b`

Weights control how much influence each input feature has on the output. Backpropagation adjusts them to minimise loss. **Biases** allow the model to output non-zero values even when all inputs are zero — they shift the activation function's threshold independently of the input, giving each neuron its own baseline firing level.

Together: **weights capture patterns; biases give flexibility in where those patterns activate.**

### 6.2 Why is an activation function required?

Without activation functions, stacking multiple layers still produces a single linear function (compositions of linear functions are linear). This means a 10-layer network without activations has no more expressive power than logistic regression.

Activation functions introduce **non-linearity**, allowing the network to learn curved, complex decision boundaries. In our experiments, `tanh` significantly outperformed `ReLU` (AUC 0.930 vs 0.746), showing that the choice of non-linearity materially affects what the model can learn.

### 6.3 What happens when learning rate is too high or too low?

**Too high (Exp 3, LR=0.01):** Large weight updates cause overshoot — the model oscillates around the minimum rather than settling into it. Recall improved dramatically (0.833) but precision collapsed (0.069) because the model overcorrected toward predicting churn for most samples.

**Too low (Exp 4, LR=0.0001):** Weight updates are tiny — the model underfits because early stopping fires before meaningful patterns are learned. AUC dropped to 0.725, barely above random.

**Optimal (0.001):** Smooth, stable convergence. The tanh model with LR=0.001 achieved AUC=0.930.

### 6.4 Did the model show signs of underfitting or overfitting?

**Underfitting was the dominant problem**, not overfitting.

| Signal | Observation |
|--------|-------------|
| Train/test accuracy gap | < 5% in all experiments — no overfitting |
| Baseline Recall = 0 | Model never predicts "churn" — complete underfit on minority class |
| Low LR model AUC = 0.725 | Stopped too early; didn't learn enough |

The root cause is the **63:1 class imbalance** — with only 25 churn examples in training, the model struggles to learn the minority class signal even with sample weighting.

The best model (Exp 5, tanh) achieves AUC=0.930 and Recall=0.833 — catching 5 of 6 churners in the test set — which is a strong result given the extreme imbalance.

**To further improve:** SMOTE oversampling, decision threshold tuning (lower from 0.5), or ensemble methods (XGBoost) would be the next steps.

---

## Evaluation Outputs

![Evaluation Outputs](results/evaluation_outputs.png)

| Panel | Description |
|-------|-------------|
| A | Confusion matrix — Baseline model |
| B | Confusion matrix — Best model (tanh) |
| C | ROC curves for all 6 experiments |
| D | AUC-ROC comparison bar chart |
| E | Recall comparison bar chart |
| F | Metric heatmap (AUC, Recall, F1, Precision) |
| G | Training loss curve — best model |
| H | Train vs test accuracy — all experiments |
| I | F1-Score comparison bar chart |

Full comparison table → `results/model_comparison_table.csv`

---

## 🛠️ Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.1.0
scipy>=1.9.0
jupyter>=1.0.0
```
