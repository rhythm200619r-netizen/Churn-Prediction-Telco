# Enterprise Churn Prediction System v3.0

> A production-ready machine learning solution for predicting customer churn with explainability and business-driven recommendations.

## Overview

This project delivers an end-to-end churn analytics pipeline combining advanced machine learning with interpretable AI. It predicts three customer risk states and provides actionable insights to minimize revenue loss through targeted interventions.

### Key Features

- **Multi-Class Churn Prediction**: Classifies customers into Safe, High Risk (< 6 months), and Medium Risk (6-24 months) categories
- **SHAP Explainability**: Transparent risk driver analysis showing why models make predictions
- **CLV-Based Recommendations**: Financial impact calculations for targeted retention strategies
- **ROI Simulation Dashboard**: Interactive analysis of intervention effectiveness
- **Live Prediction Workflow**: Terminal-based interface for real-time customer assessment

## Project Structure

```
.
├── main.py                  # End-to-end pipeline (training, evaluation, predictions)
├── telco_churn.csv         # Telco dataset with customer churn labels
└── README.md               # Project documentation
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- `pip` package manager

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install required dependencies:
```bash
pip install pandas numpy matplotlib shap xgboost scikit-learn
```

### Data Setup

Place your telco churn dataset in the project root directory with the filename `telco_churn.csv`. The CSV should include the following columns:

**Required Columns:**
- `Churn` - Target variable (Yes/No)
- `tenure` - Customer tenure in months
- `MonthlyCharges` - Monthly service charges
- `TotalCharges` - Total charges to date
- `Contract` - Contract type (Month-to-month, One year, Two year)
- `InternetService` - Service type (No, DSL, Fiber optic)
- `PaymentMethod` - Payment method
- `gender`, `Partner`, `Dependents` - Demographics
- `PhoneService`, `PaperlessBilling`, `MultipleLines` - Service flags
- `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport` - Add-on services
- `StreamingTV`, `StreamingMovies` - Streaming services

## Usage

Run the complete pipeline:
```bash
python main.py
```

### Pipeline Stages

1. **Data Cleaning** - Handles missing values, encoding, and normalization
2. **Feature Engineering** - Creates derived features (usage trends, service counts, expense ratios)
3. **Model Training** - Trains XGBoost classifier on multi-class churn windows
4. **Model Evaluation** - Classification metrics and SHAP analysis
5. **Interactive Predictions** - Terminal UI for real-time predictions on new customers

## Model Details

### Architecture
- **Algorithm**: XGBoost Classifier
- **Classes**: 
  - `0` - Safe (no churn risk)
  - `1` - High Risk (churn likely within 6 months)
  - `2` - Medium Risk (churn likely within 6-24 months)

### Feature Engineering
- **Synthetic Features**: Usage trend indicators
- **Derived Metrics**: Charges per month, service adoption count
- **Risk Flags**: High bill + new customer detection

### Performance
The model is evaluated using:
- Precision, Recall, F1-score per class
- SHAP feature importance analysis
- CLV-weighted business metrics

## Output & Results

The script generates:
- **Classification Report**: Per-class performance metrics
- **SHAP Analysis**: Feature importance and decision explanations
- **Interactive Mode**: Real-time predictions with risk assessment
- **Intervention Recommendations**: Customer-specific retention strategies

## Technologies Used

- **Pandas** - Data manipulation and cleaning
- **NumPy** - Numerical computation
- **XGBoost** - Gradient boosting classifier
- **SHAP** - Explainable AI and feature importance
- **scikit-learn** - Model utilities and preprocessing
- **Matplotlib** - Visualizations

## Error Handling

- Missing `telco_churn.csv` triggers an error with clear feedback
- Dependency validation with installation instructions
- Graceful handling of data type conversions

## Project Context

This is a final project demonstrating:
- End-to-end ML pipeline development
- Feature engineering for time-series risk windows
- Explainable AI implementation
- Business-oriented model evaluation
- Interactive user workflows

## Author

Created as a comprehensive machine learning capstone project.

## License

This project is provided as-is for educational and commercial use.

---

**Last Updated**: May 2, 2026

## How It Works

### 1. Data Cleaning

- Converts `TotalCharges` to numeric and fills invalid values with `0`
- Converts `Churn` from `Yes/No` to `1/0`
- Label-encodes binary and selected categorical fields

### 2. Feature Engineering

- Builds synthetic `usage_trend`
- Creates derived features:
  - `charges_per_month`
  - `service_count`
  - `high_bill_new`
- Creates `churn_window` target:
  - `0`: Safe
  - `1`: High Risk (`tenure <= 6` and churned)
  - `2`: Medium Risk (`6 < tenure <= 24` and churned)

### 3. Model Training

- Trains an `XGBClassifier` with train/test split and stratification
- Prints a classification report

### 4. Explainability

- Initializes `shap.TreeExplainer`
- Shows top SHAP drivers per individual prediction
- Optional SHAP waterfall chart per interactive prediction

### 5. Business Layer

- Computes customer CLV estimate
- Recommends retention actions by risk level and budget constraints
- Simulates ROI comparison:
  - No Action
  - Save Everyone
  - AI Strategy

## Run

From the project folder:

```bash
python main.py
```

Execution flow:

1. Loads and cleans data
2. Trains model
3. Displays ROI + feature-importance dashboard (matplotlib window)
4. Starts interactive terminal prediction loop

Type `exit` when prompted for tenure to quit the live system.

## Interactive Inputs

The live predictor asks for:

- Tenure
- Monthly bill
- Total charges
- Contract type (`0/1/2`)
- Internet service (`0/1/2`)
- Service count (`0-7`)
- Last month usage and this month usage

It then outputs:

- Risk class and churn probability
- CLV estimate
- Top SHAP feature impacts
- Recommended intervention strategy

## Troubleshooting

- `FileNotFoundError: telco_churn.csv`
  - Ensure `telco_churn.csv` is in the same directory as `main.py`.
- `ModuleNotFoundError`
  - Install the dependencies listed above.
- Plot window not showing
  - Check that your Python environment supports GUI backends for matplotlib.

## Notes

- `usage_trend` is partially synthetic for demonstration purposes.
- ROI simulation uses a simple random success assumption and is intended as a business illustration, not a production financial model.