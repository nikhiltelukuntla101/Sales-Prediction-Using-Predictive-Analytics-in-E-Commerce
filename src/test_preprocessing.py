from data_preprocessing import load_data, preprocess_data, save_preprocessed_data

# Step 1: Load the dataset
df = load_data('../data/ecommerce_sales_analysis.csv')  # adjust path if needed

# Step 2: Preprocess the data
df = preprocess_data(df)

# Step 3: Save the cleaned data
save_preprocessed_data(df, '../data/preprocessed_data.csv')

print("✅ Data preprocessing completed and saved.")
