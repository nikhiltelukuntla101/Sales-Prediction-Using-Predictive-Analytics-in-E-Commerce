import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_data(df):
    df = df.drop(['product_id', 'product_name'], axis=1)

    le = LabelEncoder()
    df['category'] = le.fit_transform(df['category'])

    # Save label encoder
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(le, os.path.join(model_dir, 'label_encoder.pkl'))

    # Add total sales column
    monthly_cols = [f'sales_month_{i}' for i in range(1, 13)]
    df['total_sales'] = df[monthly_cols].sum(axis=1)

    return df

if __name__ == "__main__":
    # Load and preprocess
    df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'ecommerce_sales_analysis.csv')))
    df_processed = preprocess_data(df)

    # Save preprocessed data
    df_processed.to_csv(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'preprocessed_data.csv')), index=False)
    print("✅ Preprocessing complete.")
