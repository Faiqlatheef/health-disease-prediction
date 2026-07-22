# Health Disease Prediction Pipeline

> Senior Machine Learning Engineer Take-Home Assignment

## Project Overview

This project develops an end-to-end machine learning pipeline to predict whether a patient has a disease using the provided Patient Health Record dataset.

The solution demonstrates the complete machine learning workflow including:

- Exploratory Data Analysis (EDA)
- Data Quality Assessment
- Data Preprocessing
- Feature Engineering
- Cross Validation
- Machine Learning Model Training
- Model Comparison
- Hyperparameter Tuning
- Model Evaluation
- Inference Pipeline
- Reproducible Deployment

The project follows best practices for production-ready machine learning systems using Scikit-Learn Pipelines.

---

# Problem Statement

The objective is to predict the target variable:

```
Has_Disease
```

which is a binary classification problem.

Input:

Patient health records containing demographic information, medical history, laboratory measurements and lifestyle attributes.

Output:

```
0 → No Disease
1 → Disease
```

---

# Dataset

Dataset Used

```
Patient_Health_Records_Raw.csv
```
During preprocessing, the raw dataset is cleaned, validated, transformed, and saved as:
```
clean_dataset.csv
```

Target Column

```
Has_Disease
```

---

# Repository Structure

```
health-disease-prediction/

├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── Patient_Health_Records_Raw.csv
│   └── README.md
│
├── notebooks/
│   ├── 01_visualization.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   └── 04_inference.ipynb
│
├── artifacts/
│   ├── best_pipeline.joblib
│   ├── clean_dataset.csv
│   ├── preprocessor.joblib
│   ├── feature_names.joblib
│   ├── predictions.csv
│   └── metrics.json
│
├── reports/
│   ├── figures/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── feature_importance.csv
│   └── model_comparison.csv
│
└── tests/
    └── test_pipeline.py
```

---

# Project Workflow

## Step 1

Exploratory Data Analysis

Performed:

- Dataset overview
- Missing value analysis
- Duplicate analysis
- Data types
- Statistical summary
- Outlier analysis
- Correlation analysis
- Target distribution

Generated visualizations include:

- Disease distribution
- Correlation Heatmap
- Age Distribution
- BMI Distribution
- Blood Pressure Distribution
- Cholesterol Distribution

Each visualization includes interpretation.

---

## Step 2

Data Preprocessing

The preprocessing pipeline includes:

- Missing value imputation
- Duplicate removal
- Label Encoding
- One Hot Encoding
- Feature Scaling
- Train Test Split
- Leakage Prevention using Pipeline
- ColumnTransformer

---

## Step 3

Model Training

Four machine learning algorithms are compared.

### Logistic Regression

Advantages

- Fast
- Interpretable
- Good baseline

---

### Random Forest

Advantages

- Handles nonlinear relationships
- Robust
- Feature importance

---

### Gradient Boosting

Advantages

- High predictive performance
- Handles complex interactions

---

### XGBoost (Optional)

Advantages

- State-of-the-art boosting
- Better generalization

---

Cross Validation

```
5 Fold Stratified Cross Validation
```

Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC
- Confusion Matrix

---

# Model Selection

The final model is selected based on:

- Highest ROC-AUC
- Best F1 Score
- Cross Validation Stability
- Generalization Performance

The selected model is exported using Joblib.

```
artifacts/best_pipeline.joblib
```

---

# Inference Pipeline

The inference notebook demonstrates:

- Loading trained pipeline
- Reading unseen patient records
- Data preprocessing
- Prediction
- Probability estimation

Example Output

```
Patient ID : 1204

Prediction :

Disease

Probability :

96.4%
```

---

# Results

The repository includes

- Cross Validation Scores
- Model Comparison Table
- Confusion Matrix
- ROC Curve
- Feature Importance
- Classification Report

---

# Assumptions

- Dataset is cleaned.
- Target labels are correct.
- Missing values are handled during preprocessing.
- No data leakage is introduced.

---

# Limitations

- Dataset size may limit generalization.
- No external validation dataset.
- Deep Learning models are intentionally excluded.

---

# Installation

Clone repository

```bash
git clone https://github.com/Faiqlatheef/health-disease-prediction.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Visualization

1. Open Jupyter Notebook or VS Code.

2. Execute the notebooks in the following order:

01_visualization.ipynb
02_preprocessing.ipynb
03_training.ipynb
04_inference.ipynb

## Running Tests

The project includes automated tests to verify that the trained pipeline can be loaded and used for inference.

Run:

```bash
pytest tests/
```
Expected output:
```
3 passed
```

This is a strong addition because your tests now pass successfully.

---

### 5. Results

Rather than saying only "The repository includes", mention the actual deliverables:

```markdown

## Results

The project produces:

- Cleaned dataset
- Trained machine learning pipeline
- Cross-validation metrics
- Model comparison report
- Confusion matrix
- ROC curve
- Feature importance analysis
- Predictions on unseen data

---

# Libraries Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn
- XGBoost
- Joblib

---

# Author

Senior Machine Learning Engineer Assignment

Prepared by

**Abdul Latheef Faiq Ahamed**
