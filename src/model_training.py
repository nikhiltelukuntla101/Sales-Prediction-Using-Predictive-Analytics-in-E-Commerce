import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def load_data(path):
    return pd.read_csv(path)

def train_models(df):
    # Features and target
    X = df.drop(['total_sales'], axis=1)
    y = df['total_sales']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    # Evaluation
    print("📈 Linear Regression:")
    print("MSE:", mean_squared_error(y_test, y_pred_lr))
    print("R²:", r2_score(y_test, y_pred_lr))

    print("\n🌳 Random Forest:")
    print("MSE:", mean_squared_error(y_test, y_pred_rf))
    print("R²:", r2_score(y_test, y_pred_rf))

    # Save models
    joblib.dump(lr_model, '../models/linear_regression_model.pkl')
    joblib.dump(rf_model, '../models/random_forest_model.pkl')

    print("\n✅ Models trained and saved successfully!")

# Optional: run directly
if __name__ == "__main__":
    df = load_data('../data/preprocessed_data.csv')
    train_models(df)
