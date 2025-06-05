import streamlit as st
import pandas as pd
import joblib

# Load trained models
lr_model = joblib.load('../models/linear_regression_model.pkl')
rf_model = joblib.load('../models/random_forest_model.pkl')

# App Configuration
st.set_page_config(page_title="📊 E-commerce Sales Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4B8BBE;'>🛍️ E-commerce Sales Forecast App</h1>", unsafe_allow_html=True)
st.write("Use machine learning models to predict total annual sales based on product and monthly performance data.")

# Sidebar Inputs
st.sidebar.header("📋 Enter Product Features")

category = st.sidebar.selectbox("Product Category", ['Fashion', 'Electronics', 'Grocery'])
category_map = {'Fashion': 0, 'Electronics': 1, 'Grocery': 2}
category_encoded = category_map[category]

price = st.sidebar.number_input("Price (₹)", min_value=0.0, value=500.0, step=10.0)

st.sidebar.markdown("### 🗓️ Monthly Sales")
monthly_sales = []
for i in range(1, 13):
    sale = st.sidebar.number_input(f"Month {i}", min_value=0, step=1)
    monthly_sales.append(sale)

# Predict Button
if st.button("📈 Predict Annual Sales"):

    # Prepare DataFrame
    input_data = pd.DataFrame([{
        "category": category_encoded,
        "price": price,
        **{f"sales_month_{i+1}": monthly_sales[i] for i in range(12)}
    }])

    # Model Predictions
    pred_lr = lr_model.predict(input_data)[0]
    pred_rf = rf_model.predict(input_data)[0]

    # Display Results
    col1, col2 = st.columns(2)
    col1.metric("🔹 Linear Regression Prediction", f"{pred_lr:.2f} Units")
    col2.metric("🌳 Random Forest Prediction", f"{pred_rf:.2f} Units")

    st.success("✅ Predictions generated successfully.")
    st.info("Tip: Adjust category, price, or monthly sales to test model response.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Created by Sai Nikhil, Rahul & Rushender | Powered by Machine Learning 🔍</p>", unsafe_allow_html=True)
