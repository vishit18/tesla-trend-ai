import os
import re
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from nltk import ngrams
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ensure required folders exist
os.makedirs('data', exist_ok=True)
os.makedirs('reports', exist_ok=True)

# Load raw tweets
df = pd.read_csv('data/raw/tesla_tweets_real.csv')
print(f"Loaded {len(df)} tweets.")

# Stop words that add little meaning
stop_words = set([
    'for', 'the', 'to', 'and', 'a', 'on', 'you', 'your', 'is', 'it', 'of', 'in',
    'this', 'that', 'with', 'as', 'at', 'are', 'be', 'was', 'i', 'have', 'has',
    'but', 'not', 'they', 'we', 'our', 'my', 'so', 'if', 'or', 'from', 'by',
    'an', 'will', 'just', 'all', 'about', 'out', 'up', 'what', 'can', 'me',
    'no', 'get', 'like', 'more', 'some', 'when', 'would', 'do', 'does', 'how',
    'them', 'than'
])

# Common spam/promo words to filter out tweets with giveaways, ads, etc.
spam_keywords = [
    'giveaway', 'follow', 'congratulation', 'referral', 'package',
    'dm', 'directed', 'collect', 'signed', 'contest', 'win', 'prize', 'free',
    'promo', 'promotion', 'deal', 'click', 'subscribe'
]

def clean_text(text):
    """Clean tweet text by removing URLs, mentions, hashtags, punctuation, and stop words."""
    text = re.sub(r"http\S+", "", text)       # Remove URLs
    text = re.sub(r"@\w+", "", text)          # Remove mentions
    text = re.sub(r"#", "", text)              # Remove hashtag symbols
    text = re.sub(r"[^A-Za-z\s]", "", text)   # Remove punctuation and numbers
    words = text.lower().strip().split()
    meaningful_words = [w for w in words if w not in stop_words]
    return " ".join(meaningful_words)

def is_spam(text):
    """Detect if tweet contains spam keywords or too many URLs/hashtags."""
    # Spam if contains any spam keyword
    if any(word in text for word in spam_keywords):
        return True
    # Spam if >30% of characters are non-alphabetic (sign of spammy URL/emoji)
    non_alpha_ratio = sum(1 for c in text if not c.isalpha() and not c.isspace()) / max(len(text),1)
    if non_alpha_ratio > 0.3:
        return True
    # Spam if text is too short (<5 words)
    if len(text.split()) < 5:
        return True
    return False

# Clean tweets
df['clean_text'] = df['text'].apply(clean_text)

# Filter out spammy tweets
df = df[~df['clean_text'].apply(is_spam)].reset_index(drop=True)
print(f"After spam filtering, {len(df)} tweets remain.")

# Extract n-grams (bigrams and trigrams)
def extract_ngrams(text, n):
    tokens = text.split()
    return list(ngrams(tokens, n))

all_bigrams = []
all_trigrams = []

for text in df['clean_text']:
    all_bigrams.extend(extract_ngrams(text, 2))
    all_trigrams.extend(extract_ngrams(text, 3))

bigram_freq = Counter(all_bigrams)
trigram_freq = Counter(all_trigrams)

print("\nTop 15 Bigrams:")
for grams, count in bigram_freq.most_common(15):
    print(" ".join(grams), ":", count)

print("\nTop 15 Trigrams:")
for grams, count in trigram_freq.most_common(15):
    print(" ".join(grams), ":", count)

# Sentiment analysis
analyzer = SentimentIntensityAnalyzer()
df['sentiment_score'] = df['clean_text'].apply(lambda text: analyzer.polarity_scores(text)['compound'])

print("\nSample sentiment scores:")
print(df[['clean_text', 'sentiment_score']].head())

# Save cleaned tweets with sentiment
df.to_csv('data/cleaned_tweets.csv', index=False)
print("\nSaved cleaned tweets to: data/cleaned_tweets.csv")

# Plot word frequency (top 15 single words)
all_words = " ".join(df['clean_text']).split()
word_freq = Counter(all_words)
top_words = word_freq.most_common(15)
words, counts = zip(*top_words)

plt.figure(figsize=(12,6))
plt.bar(words, counts, color='teal')
plt.title('Top 15 Words in Tesla Tweets (Filtered)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('reports/word_frequency.png')
plt.show()

# Plot top bigrams
top_bigrams = bigram_freq.most_common(15)
bigrams, bigram_counts = zip(*top_bigrams)
bigram_labels = [" ".join(b) for b in bigrams]

plt.figure(figsize=(12,6))
plt.bar(bigram_labels, bigram_counts, color='coral')
plt.title('Top 15 Bigrams in Tesla Tweets')
plt.xticks(rotation=60)
plt.tight_layout()
plt.savefig('reports/bigram_frequency.png')
plt.show()

# Plot sentiment distribution
plt.figure(figsize=(10,5))
plt.hist(df['sentiment_score'], bins=30, color='orange', edgecolor='black')
plt.title('Sentiment Score Distribution for Tesla Tweets')
plt.xlabel('Sentiment Score (Compound)')
plt.ylabel('Number of Tweets')
plt.tight_layout()
plt.savefig('reports/sentiment_distribution.png')
plt.show()
