import tweepy
import pandas as pd

import os

bearer_token = os.getenv("BEARER_TOKEN")

if bearer_token is None:
    raise ValueError("Bearer token not found. Please set the BEARER_TOKEN environment variable.")

# Initialize client
client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)

# Query parameters
query = 'Tesla -is:retweet lang:en'

# Collect tweets (max_results max is 100 per request, use pagination)
tweets = []
response = client.search_recent_tweets(query=query, tweet_fields=['created_at'], max_results=100)

if response.data:
    for tweet in response.data:
        tweets.append({
            'tweet_id': tweet.id,
            'date': tweet.created_at,
            'text': tweet.text
        })

# Save to CSV
df = pd.DataFrame(tweets)
df.to_csv('../data/tesla_tweets_real.csv', index=False)

print(f"Collected and saved {len(tweets)} tweets!")
