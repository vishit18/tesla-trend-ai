# TeslaTrend AI

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)
[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](#)

---

## Multi-source Tesla stock price prediction combining X sentiment from posts and historical prices

---

## Project Overview

This project tackles the complex challenge of merging noisy, limited-volume Twitter sentiment data collected via a free-tier API with Tesla’s historical stock price CSV data from Yahoo Finance, enabling next-day stock price prediction. Unlike typical stock price prediction projects, it handles real-world data constraints and asynchronous multi-source integration.

---

## The Problem

Stock price prediction is complicated by the need to combine structured financial time series with unstructured, noisy social media sentiment, especially when data access is limited (e.g., free Twitter API tier) leading to sparse tweet volumes and asynchronous timestamps. Most existing Tesla prediction projects ignore this integration challenge and data limitations.

---

## The Solution

- Collected Tesla-related tweets using the free-tier Twitter API, working with limited tweet volume and noisy text including slang and sarcasm. 
- Cleaned and aggregated daily tweet sentiment scores despite sparse and asynchronous timestamps.  
- Downloaded Tesla stock price data from Yahoo Finance.  
- Merged sentiment scores with stock prices by date.  
- Built a linear regression model predicting next-day closing prices based on sentiment and previous stock prices.  
- Modularized scripts for data fetching, cleaning, merging, and modeling to enable reproducibility.

---

## Business Impact

- Demonstrates the power of alternative data sources beyond traditional financial indicators. 
- Shows ability to design solutions under data volume and access constraints (free API limits). 
- Relevant for hedge funds, quant traders, and financial analysts interested in sentiment-driven strategies.  
- Highlights key skills: handling messy real-world data, building predictive models, and communicating business value clearly.  
- Shows readiness for professional ML engineering roles dealing with unstructured and structured data.

---

## Project Structure
.
├── data/
│   ├── cleaned_tweets.csv
│   ├── tesla_stock_prices.csv
│   └── merged_tesla_data.csv
├── models/
│   └── tesla_stock_model.pkl
├── src/
│   ├── get_tesla_stock.py
│   ├── clean_and_explore.py
│   ├── merge_data.py
│   ├── model_tesla_stock.py
│   └── predict.py
├── README.md
└── requirements.txt


---

## Setup and Usage

1. Create and activate a virtual environment:

    For Windows:  
    ```
    python -m venv venv
    venv\Scripts\activate
    ```

    For macOS/Linux:  
    ```
    python3 -m venv venv
    source venv/bin/activate
    ```

2. Install dependencies:  
    ```
    pip install -r requirements.txt
    ```

3. Download Tesla stock data:  
    ```
    python src/get_tesla_stock.py
    ```

4. Clean and preprocess tweets:  
    ```
    python src/clean_and_explore.py
    ```

5. Merge stock and sentiment data:  
    ```
    python src/merge_data.py
    ```

6. Train and save the prediction model:  
    ```
    python src/model_tesla_stock.py
    ```

7. Run the automated prediction:  
    ```bash
    python src/predict.py
    ```

---

## Dependencies and Installation

This project requires several Python libraries to run smoothly. To simplify setup, all required packages are listed in the requirements.txt file.

The main dependencies include:
- pandas
- matplotlib
- nltk
- vaderSentiment
- yfinance
- scikit-learn
- joblib
- numpy
- tweepy

---

## How to install dependencies

After activating your virtual environment (see Setup and Usage), install all required packages at once by running:

 pip install -r requirements.txt

This command will install the exact versions of libraries needed for TeslaTrend AI to work correctly and reproducibly.

---

## Getting Your X Bearer Token

This project uses the X API (v2) and requires a Bearer Token to access Tesla-related tweets. To use it, you must get your own key — sharing keys publicly is unsafe and not allowed.
How to get your key:
1. Visit X Developer Portal and sign in.
2. Apply for a free developer account by answering some basic questions.
3. Once approved, create a new project and app.
4. In your app dashboard, find the Keys and Tokens section.
5. Copy your key.
6. On your computer, set this key as an environment variable:
- On macOS/Linux terminal:
  ```bash
  export BEARER_TOKEN="your_key_here"

- On Windows Command Prompt:
  ```
  set BEARER_TOKEN=your_key_here

7. Run the Python scripts from the same terminal where you set the key.

Remember: Keep your API key private and don’t share it publicly.

---


## Results

- The linear regression model captures sentiment’s influence on stock price movement.  
- Model evaluation using Mean Squared Error (MSE) demonstrates prediction accuracy on a test set.  
- Saved model can be loaded for future inference or integration.

---

## Why This Project?

- Real, messy data instead of sanitized Kaggle sets — shows ability to handle practical challenges.  
- Goes beyond typical Tesla stock prediction projects by integrating multi-source data under realistic constraints (free API limits and noisy social media data). 
- Demonstrates skill in merging heterogeneous data types and managing data sparsity.
- Shows practical expertise in handling real-world data engineering and ML challenges.

---

## Future Work

- Incorporate advanced time series models (LSTMs, Transformers).  
- Automate data pipeline and retraining with CI/CD.  
- Deploy prediction model as API or interactive dashboard.  
- Extend to multiple stocks and multilingual sentiment analysis.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Contact

Questions, suggestions, or collaborations welcome.

---

*Built by Vishit Jiwane*  
