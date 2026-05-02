import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import importlib
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

try:
    XGBClassifier = importlib.import_module('xgboost').XGBClassifier
except ModuleNotFoundError:
    print("Error: Missing dependency 'xgboost'. Install it with: pip install xgboost")
    sys.exit(1)

# ==========================================
# 1. SETUP & DATA CLEANING
# ==========================================
print("=" * 55)
print("   ENTERPRISE CHURN PREDICTION SYSTEM  v3.0")
print("   Powered by XGBoost + SHAP + CLV Engine")
print("=" * 55)
print("\n--- 1. LOADING & CLEANING DATA ---")

try:
    df = pd.read_csv('telco_churn.csv')
except FileNotFoundError:
    print("Error: CSV file not found!")
    exit()

# --- Cleaning ---
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# --- Encode all binary/categorical columns XGBoost can use directly ---
binary_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'PaperlessBilling', 'MultipleLines', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies'
]
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# Encode contract and internet service as ordinal
contract_map  = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
internet_map  = {'No': 0, 'DSL': 1, 'Fiber optic': 2}
payment_map   = {
    'Electronic check': 0, 'Mailed check': 1,
    'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3
}
df['Contract']        = df['Contract'].map(contract_map).fillna(0)
df['InternetService'] = df['InternetService'].map(internet_map).fillna(0)
df['PaymentMethod']   = df['PaymentMethod'].map(payment_map).fillna(0)

print("Data Cleaned.")

# ==========================================
# 2. FEATURE ENGINEERING
# ==========================================
print("\n--- 2. FEATURE ENGINEERING ---")
np.random.seed(42)

# Synthetic usage_trend
df['usage_trend']   = np.random.normal(0, 0.05, df.shape[0])
churn_idx           = df[df['Churn'] == 1].index
high_risk_idx       = np.random.choice(churn_idx, size=int(len(churn_idx)*0.3), replace=False)
medium_risk_idx     = list(set(churn_idx) - set(high_risk_idx))
df.loc[high_risk_idx,   'usage_trend'] = np.random.uniform(-0.9, -0.5, size=len(high_risk_idx))
df.loc[medium_risk_idx, 'usage_trend'] = np.random.uniform(-0.5, -0.2, size=len(medium_risk_idx))

# Derived features
df['charges_per_month'] = df['TotalCharges'] / (df['tenure'] + 1)
df['service_count']     = (
    df['PhoneService'] + df['OnlineSecurity'] + df['OnlineBackup'] +
    df['DeviceProtection'] + df['TechSupport'] +
    df['StreamingTV'] + df['StreamingMovies']
)
df['high_bill_new'] = ((df['MonthlyCharges'] > 70) & (df['tenure'] < 12)).astype(int)

# --- Churn Window Target (multi-class) ---
def get_window(row):
    if row['Churn'] == 0:   return 0
    if row['tenure'] <= 6:  return 1
    if row['tenure'] <= 24: return 2
    return 0

df['churn_window'] = df.apply(get_window, axis=1)
print("Features Engineered.")

# ==========================================
# 3. MODEL TRAINING -- XGBoost
# ==========================================
print("\n--- 3. TRAINING XGBoost MODEL ---")

features = [
    'tenure', 'MonthlyCharges', 'TotalCharges', 'usage_trend',
    'Contract', 'InternetService', 'PaymentMethod',
    'charges_per_month', 'service_count', 'high_bill_new',
    'gender', 'Partner', 'Dependents', 'PaperlessBilling'
]

X = df[features]
y = df['churn_window']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = XGBClassifier(
    n_estimators      = 300,
    learning_rate     = 0.05,
    max_depth         = 5,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    use_label_encoder = False,
    eval_metric       = 'mlogloss',
    random_state      = 42,
    n_jobs            = -1
)
model.fit(
    X_train, y_train,
    eval_set  = [(X_test, y_test)],
    verbose   = False
)

y_pred = model.predict(X_test)
print("XGBoost Model Trained.\n")
print("Model Performance Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1, 2],
    target_names=['Safe', 'High Risk', 'Medium Risk'],
    zero_division=0
))

# ==========================================
# 4. SHAP EXPLAINER
# ==========================================
print("--- 4. SETTING UP SHAP EXPLAINER ---")
explainer = shap.TreeExplainer(model)
print("SHAP Explainer Ready.")


def get_shap_vector_for_class(shap_values, pred_class):
    # Support SHAP list output (older versions) and ndarray output (newer versions).
    if isinstance(shap_values, list):
        class_idx = min(pred_class, len(shap_values) - 1)
        return shap_values[class_idx][0], class_idx

    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            class_idx = min(pred_class, shap_values.shape[2] - 1)
            return shap_values[0, :, class_idx], class_idx
        if shap_values.ndim == 2:
            return shap_values[0], 0

    raise ValueError(f"Unsupported SHAP output type/shape: {type(shap_values)}")


def get_expected_base_value(explainer_obj, class_idx):
    expected = explainer_obj.expected_value
    if hasattr(expected, '__len__') and not np.isscalar(expected):
        return expected[class_idx]
    return expected

# ==========================================
# 5. CLV + DYNAMIC INTERVENTION ENGINE
# ==========================================

FEATURE_LABELS = {
    'tenure':            'Tenure (months)',
    'MonthlyCharges':    'Monthly Bill ($)',
    'TotalCharges':      'Total Charges ($)',
    'usage_trend':       'Usage Trend',
    'Contract':          'Contract Type',
    'InternetService':   'Internet Service',
    'PaymentMethod':     'Payment Method',
    'charges_per_month': 'Avg Charge/Month',
    'service_count':     'Number of Services',
    'high_bill_new':     'High Bill + New Customer',
    'gender':            'Gender',
    'Partner':           'Has Partner',
    'Dependents':        'Has Dependents',
    'PaperlessBilling':  'Paperless Billing',
}

def calculate_clv(monthly, tenure):
    return monthly * 12 * (1 + tenure / 60)

def recommend_intervention(pred_class, risk_pct, clv, monthly):
    max_budget = clv * 0.15

    if pred_class == 1:
        discount_val = min(monthly * 0.20, max_budget)
        return (
            f"HIGH RISK -- Immediate Retention Action\n"
            f"   Offer  : 20% Discount + 1 Free Month\n"
            f"   Budget : Max ${discount_val:,.2f}/mo authorised\n"
            f"   Channel: Retention team call within 24 hrs\n"
            f"   Urgency: Act within 48 hrs or risk losing them"
        )
    elif pred_class == 2:
        if clv > 3000:
            pct, extra = 0.15, " + Free Premium Upgrade (3 months)"
        elif clv > 1000:
            pct, extra = 0.10, " + Loyalty Points Bonus"
        else:
            pct, extra = 0.05, ""
        discount_val = min(monthly * pct, max_budget)
        return (
            f"MEDIUM RISK -- Proactive Outreach\n"
            f"   Offer  : {pct*100:.0f}% Loyalty Discount{extra}\n"
            f"   Budget : Max ${discount_val:,.2f}/mo authorised\n"
            f"   Channel: Personalised email within 72 hrs"
        )
    else:
        if risk_pct > 30:
            return (
                "WATCHLIST -- Monitor Only\n"
                "   Offer  : None yet -- flag for monthly review\n"
                "   Channel: Automated dashboard alert"
            )
        return (
            "SAFE -- No Action Needed\n"
            "   Offer  : Routine satisfaction survey in 90 days\n"
            "   Channel: Automated nurture email sequence"
        )

# ==========================================
# 6. ROI GRAPH
# ==========================================
print("\n--- 5. GENERATING ROI COMPARISON GRAPH ---")
print("Close the graph window to start the prediction system...")

def calculate_roi(strategy):
    revenue = cost = 0
    probs = model.predict_proba(X_test)
    for i in range(len(X_test)):
        risk   = probs[i][1] if probs.shape[1] > 1 else 0
        invest = (strategy == 'AI' and risk > 0.5) or (strategy == 'ALL')
        if invest:
            cost += 50
            if y_test.iloc[i] in [1, 2] and np.random.random() < 0.7:
                revenue += 600
    return revenue - cost

roi_ai  = calculate_roi('AI')
roi_all = calculate_roi('ALL')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Enterprise Churn AI -- Business Dashboard', fontsize=15, fontweight='bold')

# --- Bar: ROI Comparison ---
ax1        = axes[0]
strategies = ['No Action', 'Save Everyone', 'AI Strategy']
values     = [0, roi_all, roi_ai]
colors     = ['#95a5a6', '#3498db', '#2ecc71']
bars       = ax1.bar(strategies, values, color=colors, width=0.5, edgecolor='white', linewidth=1.2)
ax1.set_title('Strategy ROI Comparison', fontsize=12, fontweight='bold')
ax1.set_ylabel('Net Profit ($)', fontsize=11)
ax1.grid(axis='y', alpha=0.3)
for bar in bars:
    yval   = bar.get_height()
    offset = max(abs(yval) * 0.03, 300)
    ax1.text(bar.get_x() + bar.get_width()/2,
             yval + (offset if yval >= 0 else -offset * 3),
             f"${yval:,.0f}", ha='center', fontweight='bold', fontsize=11)

# --- Bar: Feature Importance from XGBoost ---
ax2         = axes[1]
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True).tail(8)
nice_labels = [FEATURE_LABELS.get(f, f) for f in importances.index]
bars2       = ax2.barh(nice_labels, importances.values, color='#9b59b6', edgecolor='white')
ax2.set_title('Top Feature Importances (XGBoost)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Importance Score', fontsize=11)
ax2.grid(axis='x', alpha=0.3)
for bar in bars2:
    w = bar.get_width()
    ax2.text(w + 0.002, bar.get_y() + bar.get_height()/2,
             f"{w:.3f}", va='center', fontsize=9)

plt.tight_layout()
plt.show()

# ==========================================
# 7. INTERACTIVE PREDICTION SYSTEM
# ==========================================
print("\n" + "=" * 55)
print("   LIVE CHURN PREDICTION SYSTEM  v3.0")
print("=" * 55)
print("Engine  : XGBoost (Enterprise Grade)")
print("Explain : SHAP Explainability")
print("Budget  : CLV-Based Intervention")

CONTRACT_MAP = {'0': 'Month-to-month', '1': 'One year', '2': 'Two year'}
INTERNET_MAP = {'0': 'No internet',    '1': 'DSL',      '2': 'Fiber optic'}

while True:
    print("\n" + "-" * 45)
    user_input = input("1.  Tenure (Months)           [or 'exit']: ").strip().lower()
    if user_input == 'exit':
        print("Goodbye!")
        break

    try:
        tenure  = float(user_input)
        monthly = float(input("2.  Monthly Bill ($)                      : "))
        total   = float(input("3.  Total Charges ($)                     : "))

        print("    Contract  -> 0=Month-to-month  1=One year  2=Two year")
        contract = int(input("4.  Contract Type (0/1/2)                 : "))

        print("    Internet  -> 0=None  1=DSL  2=Fiber optic")
        internet = int(input("5.  Internet Service (0/1/2)              : "))

        svc_count = int(input("6.  Number of Active Services (0-7)      : "))

        print("\n--- Usage Data ---")
        last  = float(input("    Usage Last Month (Minutes/GB)         : "))
        curr  = float(input("    Usage This Month (Minutes/GB)         : "))
        trend = (curr - last) / last if last != 0 else 0.0
        direction = "Dropping" if trend < -0.1 else "Stable/Rising"
        print(f"    >> Calculated Trend: {trend:+.2f}  ({direction})")

        # --- Derived features ---
        charges_per_month = total / (tenure + 1)
        high_bill_new     = int(monthly > 70 and tenure < 12)

        # --- Predict ---
        input_data = pd.DataFrame([[
            tenure, monthly, total, trend,
            contract, internet, 1,
            charges_per_month, svc_count, high_bill_new,
            0, 0, 0, 0
        ]], columns=features)

        pred       = model.predict(input_data)[0]
        probs      = model.predict_proba(input_data)[0]
        risk_score = sum(probs[1:])
        risk_pct   = risk_score * 100

        # --- SHAP ---
        shap_values = explainer.shap_values(input_data)
        sv, class_idx = get_shap_vector_for_class(shap_values, pred)

        shap_pairs = sorted(
            zip(features, sv, input_data.iloc[0]),
            key=lambda x: abs(x[1]), reverse=True
        )

        # --- CLV & Intervention ---
        clv        = calculate_clv(monthly, tenure)
        action_str = recommend_intervention(pred, risk_pct, clv, monthly)

        # --- Output ---
        print("\n" + "DIAGNOSIS ".ljust(45, "-"))

        if pred == 1:
            print("RESULT     : HIGH RISK  (Churn likely < 6 months)")
        elif pred == 2:
            print("RESULT     : MEDIUM RISK (Churn likely 6-24 months)")
        else:
            print(f"RESULT     : {'WATCHLIST' if risk_pct > 30 else 'SAFE CUSTOMER'}")

        print(f"Churn Prob : {risk_pct:.1f}%")
        print(f"Est. CLV   : ${clv:,.2f}   |   Annual Loss if Churned: ${monthly*12:,.2f}")
        print(f"Contract   : {CONTRACT_MAP.get(str(contract), contract)}")
        print(f"Internet   : {INTERNET_MAP.get(str(internet), internet)}")

        print("\nWHY THE AI FLAGGED THIS CUSTOMER (Top SHAP Drivers):")
        for feat, sv_val, feat_val in shap_pairs[:5]:
            arrow = "raises" if sv_val > 0 else "lowers"
            label = FEATURE_LABELS.get(feat, feat)
            bar   = "|" * int(abs(sv_val) * 100)
            print(f"   {arrow} risk  |  {label}: {feat_val:.2f}  [{bar}]  ({sv_val:+.4f})")

        print(f"\nRECOMMENDED ACTION:\n   {action_str}")

        # --- Optional SHAP waterfall ---
        show = input("\nShow SHAP waterfall chart? (y/n): ").strip().lower()
        if show == 'y':
            exp = shap.Explanation(
                values        = sv,
                base_values   = get_expected_base_value(explainer, class_idx),
                data          = input_data.iloc[0].values,
                feature_names = [FEATURE_LABELS[f] for f in features]
            )
            shap.plots.waterfall(exp, show=False)
            plt.title("SHAP Explanation -- Why this Churn Risk?", fontsize=12)
            plt.tight_layout()
            plt.show()

    except ValueError as e:
        print(f"Invalid input: {e}. Please try again.")