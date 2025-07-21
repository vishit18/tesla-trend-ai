
import yfinance as yf

# Download Tesla stock data for last 30 days, daily interval
tesla_data = yf.download('TSLA', period='1mo', interval='1d')

# Save the data to CSV for later use
tesla_data.to_csv('data/tesla_stock_prices.csv')

print("Tesla stock prices downloaded and saved.")
