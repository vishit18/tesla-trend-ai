import joblib
import numpy as np

# Load the trained model
model = joblib.load('models/tesla_stock_model.pkl')

# Function to predict next day's stock price
def predict_next_price(sentiment_score, close_price):
    input_features = np.array([[sentiment_score, close_price]])
    predicted_price = model.predict(input_features)
    return predicted_price[0]

# === Interactive part for testing ===
if __name__ == "__main__":
    try:
        sentiment_score = float(input("Enter sentiment score (e.g. 0.35): "))
        close_price = float(input("Enter today's closing stock price (e.g. 289.5): "))
        prediction = predict_next_price(sentiment_score, close_price)
        print(f"\n📈 Predicted Tesla stock price for next day: ${prediction:.2f}")
    except Exception as e:
        print("Error:", e)
