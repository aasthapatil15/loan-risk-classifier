import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.set_page_config(page_title="Loan Risk Classifier (5000 Records)", layout="wide")
st.title("🏦 Loan Approval & Risk Classifier")
st.write("A Data Mining mini-project trained on **5,000 dataset records** comparing Decision Trees and Naive Bayes.")

# 1. Dataset Generation (5,000 Records)
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 5000
    income = np.random.randint(20000, 200000, n)
    credit_score = np.random.randint(300, 850, n)
    loan_amount = np.random.randint(5000, 80000, n)
    
    # Approval rule based on credit history and debt-to-income ratio
    approved = ((credit_score >= 650) & (income >= loan_amount * 1.5)).astype(int)
    
    return pd.DataFrame({
        "Income": income,
        "CreditScore": credit_score,
        "LoanAmount": loan_amount,
        "Approved": approved
    })

df = load_data()
X = df[["Income", "CreditScore", "LoanAmount"]]
y = df["Approved"]

# 2. View & Inspect the Dataset Section
with st.expander("📁 View & Inspect Dataset (5,000 Rows)"):
    st.write("Preview of the generated dataset:")
    st.dataframe(df, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Dataset Summary Statistics:**")
        st.dataframe(df.describe())
    with col_b:
        st.write("**Class Distribution (Approved vs. Rejected):**")
        st.write(df["Approved"].value_counts().rename({1: "Approved (1)", 0: "Rejected (0)"}))
        
        # Download button to download the 5000 records as CSV
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset as CSV",
            data=csv_data,
            file_name="loan_dataset_5000.csv",
            mime="text/csv"
        )

# 3. Split: 80% Train (4,000), 20% Test (1,000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train models
dt_model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
nb_model = GaussianNB().fit(X_train, y_train)

dt_acc = accuracy_score(y_test, dt_model.predict(X_test))
nb_acc = accuracy_score(y_test, nb_model.predict(X_test))

# 5. Model Evaluation and Predictor UI
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📊 Model Performance (5,000 Samples)")
    st.info(f"Total Dataset: **{len(df):,} records** | Train: **{len(X_train):,}** | Test: **{len(X_test):,}**")
    
    st.metric(label="Decision Tree Accuracy", value=f"{dt_acc * 100:.1f}%")
    st.metric(label="Naive Bayes Accuracy", value=f"{nb_acc * 100:.1f}%")
    
    st.write("#### Confusion Matrix (Decision Tree)")
    cm = confusion_matrix(y_test, dt_model.predict(X_test))
    fig, ax = plt.subplots(figsize=(4, 3))
    ConfusionMatrixDisplay(cm, display_labels=["Rejected", "Approved"]).plot(ax=ax, cmap="Blues", colorbar=False)
    st.pyplot(fig)

with col2:
    st.subheader("🔍 Check Loan Eligibility")
    user_income = st.number_input("Annual Income (₹)", min_value=10000, max_value=500000, value=75000, step=5000)
    user_score = st.slider("Credit Score (CIBIL)", min_value=300, max_value=850, value=720)
    user_loan = st.number_input("Requested Loan Amount (₹)", min_value=2000, max_value=150000, value=30000, step=2500)
    
    selected_algo = st.selectbox("Choose Model to Test", ["Decision Tree", "Naive Bayes"])
    model = dt_model if selected_algo == "Decision Tree" else nb_model
    
    if st.button("Check Loan Status"):
        result = model.predict([[user_income, user_score, user_loan]])[0]
        if result == 1:
            st.success("✅ Application Approved! (Low Risk)")
        else:
            st.error("❌ Application Rejected! (High Risk)")
