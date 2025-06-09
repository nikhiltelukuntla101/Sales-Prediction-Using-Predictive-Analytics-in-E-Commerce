from src.data_preprocessing import load_data, preprocess_data
def test():
    df = load_data("data/ecommerce_sales_analysis.csv")
    df = preprocess_data(df)
    assert not df.isnull().values.any(), "❌ There are missing values after preprocessing!"
    print("✅ Preprocessing test passed.")

if __name__ == "__main__":
    test()
