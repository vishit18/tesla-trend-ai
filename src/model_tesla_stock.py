
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# === Load the merged tweets + stock data ===
df = pd.read_csv('data/merged_tesla_data.csv', parse_dates=['Date'])

print(f"Loaded {len(df)} rows of merged data.")

# === Data preprocessing ===

# Fill missing sentiment scores with 0 (neutral)
df['sentiment_score'] = df['sentiment_score'].fillna(0)

# Convert Close prices to float (just in case)
df['Close'] = df['Close'].astype(float)

# We want to predict the next day's Close price
df['Close_next_day'] = df['Close'].shift(-1)

# Drop the last row because it has no next day Close price
df = df.dropna(subset=['Close_next_day'])

print(f"Data after preparing target variable has {len(df)} rows.")

# === Feature selection ===
# We'll use sentiment score and today's Close price to predict next day's Close
X = df[['sentiment_score', 'Close']]
y = df['Close_next_day']

# === Train-test split ===
# 80% train, 20% test split, fixed random seed for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Model training ===
model = LinearRegression()
model.fit(X_train, y_train)

# === Prediction and evaluation ===
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on test set: {mse:.4f}")

# === Save the trained model ===
model_filename = "models/tesla_stock_model.pkl"
joblib.dump(model, model_filename)
print(f"Trained model saved to: {model_filename}")

# === Optional: Show coefficients ===
print("Model coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"  {feature}: {coef:.4f}")

print(f"Intercept: {model.intercept_:.4f}")
