import joblib
import numpy as np
import pandas as pd

def main():
    # Load model
    model = joblib.load('models/tesla_stock_model.pkl')

    # Load merged Tesla stock + sentiment data
    df = pd.read_csv('data/merged_tesla_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')

    # Select latest data row
    latest = df.iloc[-1]
    sentiment = latest['sentiment_score']
    close_price = latest['Close']

    # Prepare input features and predict
    features = np.array([[sentiment, close_price]])
    prediction = model.predict(features)[0]

    # Output results
    print(f"\nLatest date: {latest['Date'].date()}")
    print(f"Sentiment Score: {sentiment:.3f}")
    print(f"Latest Closing Price: ${close_price:.2f}")
    print(f"\nPredicted Tesla Stock Price for Next Day: ${prediction:.2f}")

if __name__ == "__main__":
    main()
