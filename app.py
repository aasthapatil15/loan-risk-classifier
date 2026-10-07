import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="AI Loan Risk & Approval Engine",
    page_icon="🏦",
    layout="wide"
)

# Custom header styling
st.markdown("""
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1rem;
        color: #64748B;
        margin-bottom: 25px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏦 AI Loan Risk & Eligibility Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Data Mining & Warehousing Mini-Project | 5,000 Records | Decision Tree vs. Naive Bayes</div>', unsafe_allow_html=True)

# ----------------- 1. DATASET GENERATION -----------------
@st.cache_data
def load_data():
    np.random.seed(42)
    n = 5000
    income = np.random.randint(20000, 200000, n)
    credit_score = np.random.randint(300, 850, n)
    loan_amount = np.random.randint(5000, 80000, n)
    
    # Financial ratio logic: Approval requires healthy credit + safe debt coverage
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

# ----------------- 2. DATA INSPECTOR -----------------
with st.expander("📁 View & Inspect Dataset (5,000 Real-Time Records)"):
    tab1, tab2 = st.tabs(["Dataset Preview", "Statistical Summary"])
    with tab1:
        st.dataframe(df, use_container_width=True, height=250)
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download 5,000-Row Dataset (CSV)",
            data=csv_data,
            file_name="loan_data_5000.csv",
            mime="text/csv"
        )
    with tab2:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.write("**Feature Distribution:**")
            st.dataframe(df.describe().T[["mean", "std", "min", "max"]])
        with col_s2:
            st.write("**Target Class Distribution:**")
            counts = df["Approved"].value_counts().rename({1: "Approved", 0: "Rejected"})
            st.bar_chart(counts)

# ----------------- 3. MODEL TRAINING -----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
nb_model = GaussianNB().fit(X_train, y_train)

dt_preds = dt_model.predict(X_test)
nb_preds = nb_model.predict(X_test)

# Metrics calculation
metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
    "Decision Tree": [
        f"{accuracy_score(y_test, dt_preds):.2%}",
        f"{precision_score(y_test, dt_preds):.2%}",
        f"{recall_score(y_test, dt_preds):.2%}",
        f"{f1_score(y_test, dt_preds):.2%}"
    ],
    "Gaussian Naive Bayes": [
        f"{accuracy_score(y_test, nb_preds):.2%}",
        f"{precision_score(y_test, nb_preds):.2%}",
        f"{recall_score(y_test, nb_preds):.2%}",
        f"{f1_score(y_test, nb_preds):.2%}"
    ]
})

# ----------------- 4. DASHBOARD SPLIT -----------------
col_eval, col_pred = st.columns([1.1, 1], gap="large")

with col_eval:
    st.subheader("📊 Comparative Algorithm Benchmark")
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.write("#### Confusion Matrices (Testing on 1,000 Unseen Records)")
    cm_tab1, cm_tab2 = st.tabs(["Decision Tree", "Naive Bayes"])
    
    with cm_tab1:
        cm_dt = confusion_matrix(y_test, dt_preds)
        fig_dt, ax_dt = plt.subplots(figsize=(4, 2.8))
        ConfusionMatrixDisplay(cm_dt, display_labels=["Rejected", "Approved"]).plot(ax=ax_dt, cmap="Blues", colorbar=False)
        plt.title("Decision Tree Confusion Matrix", fontsize=10)
        st.pyplot(fig_dt)
        
    with cm_tab2:
        cm_nb = confusion_matrix(y_test, nb_preds)
        fig_nb, ax_nb = plt.subplots(figsize=(4, 2.8))
        ConfusionMatrixDisplay(cm_nb, display_labels=["Rejected", "Approved"]).plot(ax=ax_nb, cmap="Greens", colorbar=False)
        plt.title("Naive Bayes Confusion Matrix", fontsize=10)
        st.pyplot(fig_nb)

with col_pred:
    st.subheader("🔍 Interactive Applicant Predictor")
    with st.container(border=True):
        user_income = st.number_input("Annual Income (₹)", min_value=15000, max_value=500000, value=75000, step=5000)
        user_score = st.slider("CIBIL / Credit Score", min_value=300, max_value=850, value=720)
        user_loan = st.number_input("Requested Loan Amount (₹)", min_value=5000, max_value=150000, value=30000, step=2500)
        
        algo_choice = st.radio("Selected Mining Classifier", ["Decision Tree", "Gaussian Naive Bayes"], horizontal=True)
        active_model = dt_model if algo_choice == "Decision Tree" else nb_model
        
        if st.button("Evaluate Credit Risk", type="primary", use_container_width=True):
            input_vector = [[user_income, user_score, user_loan]]
            prediction = active_model.predict(input_vector)[0]
            probability = active_model.predict_proba(input_vector)[0][1]  # Chance of approval
            
            st.divider()
            if prediction == 1:
                st.success(f"### ✅ Status: Approved (Low Risk)")
                st.progress(float(probability))
                st.write(f"**Approval Confidence:** `{probability:.1%}`")
            else:
                st.error(f"### ❌ Status: Rejected (High Risk)")
                st.progress(float(probability))
                st.write(f"**Approval Confidence:** `{probability:.1%}` *(Below Threshold)*")
