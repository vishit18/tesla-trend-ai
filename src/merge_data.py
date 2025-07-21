import pandas as pd

# Paths to your cleaned tweets and stock price CSVs
tweets_path = 'data/cleaned_tweets.csv'
stock_path = 'data/tesla_stock_prices.csv'

# Load cleaned tweets data
tweets_df = pd.read_csv(tweets_path)
print(f"Loaded {len(tweets_df)} tweets.")

# Load Tesla stock data
# Skip first two rows because they contain repeated headers/info, not actual data
stock_df = pd.read_csv(stock_path, skiprows=2)

print("Columns in stock data after skipping rows:", stock_df.columns.tolist())

# Rename columns if needed (some Yahoo data formats can be inconsistent)
# Usually columns should be: Date, Close, High, Low, Open, Volume
# Let’s just confirm and rename explicitly for safety
stock_df.columns = ['Date', 'Close', 'High', 'Low', 'Open', 'Volume']

# Convert date columns to datetime objects for merging
stock_df['Date'] = pd.to_datetime(stock_df['Date'])
tweets_df['date'] = pd.to_datetime(tweets_df['date'])

print("Sample stock data after cleanup:")
print(stock_df.head())

print("Sample tweets data:")
print(tweets_df.head())

# Since tweets have time info but stock data is daily,
# we'll merge on the date part only (ignore time in tweets)
tweets_df['date_only'] = tweets_df['date'].dt.date
stock_df['date_only'] = stock_df['Date'].dt.date

# Merge tweets and stock prices on the date_only column
merged_df = pd.merge(tweets_df, stock_df, on='date_only', how='inner')

print(f"Merged data has {len(merged_df)} rows.")

print("Here’s a quick preview of the merged data:")
print(merged_df[['date', 'clean_text', 'sentiment_score', 'Close', 'Open', 'High', 'Low', 'Volume']].head())

# Save merged data for next steps (like modeling)
merged_df.to_csv('data/merged_tweets_stock.csv', index=False)
print("Merged dataset saved to data/merged_tweets_stock.csv")
