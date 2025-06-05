import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    # Drop non-numeric identifiers (not useful for prediction)
    df = df.drop(['product_id', 'product_name'], axis=1)

    # Encode 'category' using Label Encoding
    le = LabelEncoder()
    df['category'] = le.fit_transform(df['category'])

    # Create total sales feature (sum of monthly sales)
    monthly_cols = [f'sales_month_{i}' for i in range(1, 13)]
    df['total_sales'] = df[monthly_cols].sum(axis=1)

    return df

def save_preprocessed_data(df, path='data/preprocessed_data.csv'):
    df.to_csv(path, index=False)
