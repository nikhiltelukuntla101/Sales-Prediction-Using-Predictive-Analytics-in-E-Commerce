import streamlit as st
import pandas as pd
import joblib
import os
import shap
import matplotlib.pyplot as plt
import numpy as np
import streamlit.components.v1 as components

# ==== Load models and encoder ====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
lr_model = joblib.load(os.path.join(BASE_DIR, '../models/linear_regression_model.pkl'))
rf_model = joblib.load(os.path.join(BASE_DIR, '../models/random_forest_model.pkl'))
label_encoder = joblib.load(os.path.join(BASE_DIR, '../models/label_encoder.pkl'))

# Load data to extract categories
df_all = pd.read_csv(os.path.join(BASE_DIR, '../data/preprocessed_data.csv'))
unique_cat_labels = sorted(df_all['category'].unique())
unique_cat_names = label_encoder.inverse_transform(unique_cat_labels)

# ==== Page Config ====
st.set_page_config(page_title="🛍️ E-commerce Sales Forecast", layout="wide")
st.title("🛍️ E-commerce Sales Forecast App")
st.markdown("Use ML models to predict annual sales based on category, price, review scores, and monthly sales.")

# ==== Sidebar Input ====
st.sidebar.header("📋 Product Details")

category_name = st.sidebar.selectbox("Select Category", unique_cat_names)
category_encoded = label_encoder.transform([category_name])[0]
price = st.sidebar.number_input("Price (₹)", min_value=0.0, value=500.0, step=10.0)
review_score = st.sidebar.slider("Average Review Score", 1.0, 5.0, 4.0, step=0.1)
review_count = st.sidebar.number_input("Number of Reviews", min_value=0, value=100)

st.sidebar.markdown("### 📆 Monthly Sales")
monthly_sales = [st.sidebar.number_input(f"Month {i}", min_value=0, step=1) for i in range(1, 13)]

# ==== Toggle Model Choice ====
st.sidebar.markdown("### 🧠 Choose Model")
selected_model_name = st.sidebar.radio("Model", ["Linear Regression", "Random Forest"])
model = lr_model if selected_model_name == "Linear Regression" else rf_model

# ==== Predict Button ====
if st.button("📈 Predict Sales"):
    input_data = {
        "category": category_encoded,
        "price": price,
        "review_score": review_score,
        "review_count": review_count,
    }
    input_data.update({f"sales_month_{i+1}": monthly_sales[i] for i in range(12)})

    input_df = pd.DataFrame([input_data])
    total_monthly_sales = sum(monthly_sales)
    prediction = model.predict(input_df)[0]

    # ==== Display Results ====
    st.subheader("📊 Prediction Results")
    col1, col2 = st.columns([2, 1])
    col1.metric(f"🧠 {selected_model_name} Prediction", f"{prediction:.2f} Units")
    col2.metric("📅 Total Monthly Sales", f"{total_monthly_sales} Units")

    st.success("✅ Prediction generated successfully!")
    st.info("💡 Tip: Try adjusting input values to analyze model behavior.")

       # ==== SHAP Explanation (Only for Random Forest) ====
    if selected_model_name == "Random Forest":
        st.markdown("---")
        st.subheader("🔍 Feature Importance with SHAP")

        # Initialize SHAP
        shap.initjs()
        explainer = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(input_df)

        # Local Explanation
        st.markdown("#### 🔎 Local Explanation (Single Prediction)")
        force_plot_html = shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_df,
            matplotlib=False,
            show=False
        )
        shap_html_path = os.path.join(BASE_DIR, "force_plot.html")
        shap.save_html(shap_html_path, force_plot_html)

        with open(shap_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            components.html(html_content, height=300)

        # Global Explanation
        st.markdown("#### 🌐 Global Feature Importance")
        fig_summary, ax = plt.subplots()
        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False)
        st.pyplot(fig_summary)

# ==== Footer ====
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Made by Sai Nikhil, Rahul & Rushender | Powered by Machine Learning 🔍</p>", unsafe_allow_html=True)
