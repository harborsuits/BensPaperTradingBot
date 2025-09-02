from datetime import datetime, timedelta
import praw
import cohere 
from newsapi import NewsApiClient
import pandas as pd 
from datetime import datetime
import logging 
import pytz
import time 
import warnings 
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel
import torch
from collections import defaultdict
import numpy as np
import random
from huggingface_hub import login
warnings.filterwarnings('ignore')

class TextFetchPipeline:
    def __init__(self, news_api_key, reddit_client_id, reddit_client_secret, reddit_user_agent, cohere_key, tickers, model, tokenizer):
        # Initialize APIs
        self.news_api = NewsApiClient(api_key=news_api_key)
        self.reddit = praw.Reddit(
            client_id=reddit_client_id,
            client_secret=reddit_client_secret,
            user_agent=reddit_user_agent
        )
        
        # Cache to avoid duplicate processing
        self.news_cache = set()
        self.reddit_cache = set()
        
        # Ticker to company name mapping
        self.ticker_company_map = {
        'AAPL': "Apple Inc.",
        'MSFT': "Microsoft Corporation",
        'AMZN': "Amazon.com, Inc.",
        'GOOGL': "Alphabet Inc. (Class A)",
        'GOOG': "Alphabet Inc. (Class C)",
        'META': "Meta Platforms Inc.",
        'TSLA': "Tesla, Inc.",
        'BRK.B': "Berkshire Hathaway Inc. (Class B)",
        'JNJ': "Johnson & Johnson",
        'V': "Visa Inc.",
        'WMT': "Walmart Inc.",
        'JPM': "JPMorgan Chase & Co.",
        'PG': "Procter & Gamble Company",
        'UNH': "UnitedHealth Group Incorporated",
        'NVDA': "NVIDIA Corporation",
        'HD': "The Home Depot, Inc.",
        'DIS': "The Walt Disney Company",
        'PYPL': "PayPal Holdings, Inc.",
        'MA': "Mastercard Incorporated",
        'BAC': "Bank of America Corporation",
        'XOM': "Exxon Mobil Corporation",
        'NFLX': "Netflix, Inc.",
        'KO': "The Coca-Cola Company",
        'PFE': "Pfizer Inc.",
        'MRK': "Merck & Co., Inc.",
        'INTC': "Intel Corporation",
        'T': "AT&T Inc.",
        'CSCO': "Cisco Systems, Inc.",
        'PEP': "PepsiCo, Inc.",
        'VZ': "Verizon Communications Inc.",
        'ABT': "Abbott Laboratories",
        'CMCSA': "Comcast Corporation",
        'ADBE': "Adobe Inc.",
        'CRM': "Salesforce, Inc.",
        'NKE': "NIKE, Inc.",
        'WFC': "Wells Fargo & Company",
        'MCD': "McDonald's Corporation",
        'IBM': "International Business Machines Corporation",
        'HON': "Honeywell International Inc.",
        'ORCL': "Oracle Corporation",
        'COST': "Costco Wholesale Corporation",
        'CVX': "Chevron Corporation",
        'LLY': "Eli Lilly and Company",
        'DHR': "Danaher Corporation",
        'MDT': "Medtronic plc",
        'TXN': "Texas Instruments Incorporated",
        'UNP': "Union Pacific Corporation",
        'LIN': "Linde plc",
        'NEE': "NextEra Energy, Inc.",
        'PM': "Philip Morris International Inc.",
        'BMY': "Bristol-Myers Squibb Company",
        'QCOM': "QUALCOMM Incorporated",
        'SBUX': "Starbucks Corporation",
        'AVGO': "Broadcom Inc.",
        'GILD': "Gilead Sciences, Inc.",
        'UPS': "United Parcel Service, Inc.",
        'AXP': "American Express Company",
        'LOW': "Lowe's Companies, Inc.",
        'MS': "Morgan Stanley",
        'AMD': "Advanced Micro Devices, Inc.",
        'SPGI': "S&P Global Inc.",
        'BLK': "BlackRock, Inc.",
        'PLD': "Prologis, Inc.",
        'AMT': "American Tower Corporation",
        'SYK': "Stryker Corporation",
        'ISRG': "Intuitive Surgical, Inc.",
        'BKNG': "Booking Holdings Inc.",
        'CAT': "Caterpillar Inc.",
        'GS': "The Goldman Sachs Group, Inc.",
        'DE': "Deere & Company",
        'LMT': "Lockheed Martin Corporation",
        'GE': "General Electric Company",
        'NOW': "ServiceNow, Inc.",
        'ADI': "Analog Devices, Inc.",
        'MO': "Altria Group, Inc.",
        'SCHW': "The Charles Schwab Corporation",
        'TMO': "Thermo Fisher Scientific Inc.",
        'MMM': "3M Company",
        'AMAT': "Applied Materials, Inc.",
        'BA': "The Boeing Company",
        'LRCX': "Lam Research Corporation",
        'MDLZ': "Mondelez International, Inc.",
        'FIS': "Fidelity National Information Services, Inc.",
        'ATVI': "Activision Blizzard, Inc.",
        'MU': "Micron Technology, Inc.",
        'FDX': "FedEx Corporation",
        'SO': "The Southern Company",
        'CL': "Colgate-Palmolive Company",
        'C': "Citigroup Inc.",
        'DUK': "Duke Energy Corporation",
        'APD': "Air Products and Chemicals, Inc.",
        'BSX': "Boston Scientific Corporation",
        'TJX': "The TJX Companies, Inc.",
        'EL': "The Estée Lauder Companies Inc.",
        'GM': "General Motors Company",
        'BDX': "Becton, Dickinson and Company",
        'WM': "Waste Management, Inc.",
        'ZTS': "Zoetis Inc.",
        'CI': "Cigna Corporation",
        'CB': "Chubb Limited",
        'ICE': "Intercontinental Exchange, Inc.",
        'ADI': "Analog Devices, Inc.",
        'MMC': "Marsh & McLennan Companies, Inc.",
        'CME': "CME Group Inc.",
        'ADP': "Automatic Data Processing, Inc.",
        'PNC': "The PNC Financial Services Group, Inc.",
        'USB': "U.S. Bancorp",
        'MET': "MetLife, Inc.",
        'AON': "Aon plc",
        'TFC': "Truist Financial Corporation",
        'BK': "The Bank of New York Mellon Corporation",
        'COF': "Capital One Financial Corporation",
        'PRU': "Prudential Financial, Inc.",
        'ALL': "The Allstate Corporation",
        'AIG': "American International Group, Inc.",
        'HIG': "The Hartford Financial Services Group, Inc.",
        'TRV': "The Travelers Companies, Inc.",
        'PGR': "The Progressive Corporation",
        'LNC': "Lincoln National Corporation",
        'UNM': "Unum Group",
        'AFL': "Aflac Incorporated",
        'GL': "Globe Life Inc.",
        'L': "Loews Corporation",
        'CINF': "Cincinnati Financial Corporation",
        'WRB': "W. R. Berkley Corporation",
        'RE': "Everest Re Group, Ltd.",
        'MKL': "Markel Corporation",
        'RJF': "Raymond James Financial, Inc.",
        'STT': "State Street Corporation",
        'NTRS': "Northern Trust Corporation",
        'AMP': "Ameriprise Financial, Inc.",
        'IVZ': "Invesco Ltd.",
        'BEN': "Franklin Resources, Inc.",
        'TROW': "T. Rowe Price Group, Inc.",
        'SCHW': "The Charles Schwab Corporation",
        'ETFC': "E*TRADE Financial Corporation",
        }
 

        self.tickers = defaultdict()

        self.co = cohere.Client(cohere_key)
        self.agg_text = {}

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("finnhub_websocket.log"),
            ],
        )

        # Model and tokenizer passed from main
        self.model = model
        self.tokenizer = tokenizer
        self.sentiment = defaultdict()
        self.prob = defaultdict()

        for ticker in tickers:
            self.tickers[ticker] = self.ticker_company_map.get(ticker, "Unknown Company")
            self.sentiment[ticker] = 'Neutral'
            self.prob[ticker] = 1.0 
        

    def fetch_news(self, ticker):
        """
        Fetch latest news headlines for the ticker or company name.
        Args:
            ticker (str): Ticker symbol (e.g., "TSLA").
            company_name (str): Company name (e.g., "Tesla").
        Returns:
            list: List of new, unique news headlines with associated ticker.
        """
        company_name = self.tickers[ticker]
        now = datetime.utcnow()
        thirty_minutes_ago = now - timedelta(minutes=30)
        now_str = now.strftime('%Y-%m-%dT%H:%M:%S')
        thirty_minutes_ago_str = thirty_minutes_ago.strftime('%Y-%m-%dT%H:%M:%S')

        # Fetch articles from the News API
        articles = self.news_api.get_everything(
            q=f"{ticker} OR {company_name}",
            from_param=thirty_minutes_ago_str,
            to=now_str,
            language="en",
            sort_by="publishedAt"
        )

        new_articles = []
        for article in articles.get("articles", []):
            title = article["title"]
            published_at = datetime.strptime(article["publishedAt"], '%Y-%m-%dT%H:%M:%SZ')

            # Check if the article is within the 30-minute window and not in cache
            if title not in self.news_cache and published_at >= thirty_minutes_ago:
                self.news_cache.add(title)
                new_articles.append({"source": "news", "text": title, "ticker": ticker})

        return new_articles

    def fetch_reddit(self, ticker):
        """
        Fetch Reddit posts containing the ticker from r/wallstreetbets.
        Args:
            ticker (str): Ticker symbol (e.g., "TSLA").
        Returns:
            list: List of dictionaries with Reddit post titles and associated ticker.
        """
        company_name = self.tickers[ticker]
        reddit_posts = []
        current_time_utc = datetime.utcnow().replace(tzinfo=pytz.UTC)  # Ensure current time is in UTC
        
        # Ensure the cache contains tuples (post_id, timestamp)
        if self.reddit_cache and not all(isinstance(item, tuple) and len(item) == 2 for item in self.reddit_cache):
            self.reddit_cache = set()

        # Remove stale entries older than 10 minutes
        self.reddit_cache = {
            (post_id, timestamp) for post_id, timestamp in self.reddit_cache
            if timestamp > current_time_utc - timedelta(minutes=10)
        }

        for post in self.reddit.subreddit("wallstreetbets").new(limit=50):
            text = post.selftext.lower()  # Ensure case-insensitive matching
            post_time_utc = datetime.fromtimestamp(post.created_utc, pytz.UTC)  # Convert post time to UTC
            
            # Check cache and if the ticker/company name appears in the post title
            if (post.id, post_time_utc) not in self.reddit_cache and (ticker.lower() in text or company_name.lower() in text):
                reddit_posts.append({"source": "reddit", "text": text, "ticker": ticker})
                self.reddit_cache.add((post.id, post_time_utc))  # Add to cache

        return reddit_posts

    def fetch_combined_data(self):
        """
        Fetch data from news and Reddit for all tickers and prepare for sentiment analysis.
        Returns:
            pd.DataFrame: Combined data with columns 'source', 'text', and 'ticker'.
        """
        combined_data = []
        for ticker in self.tickers.keys():
            # Fetch news and Reddit data for each ticker
            news_data = self.fetch_news(ticker)
            reddit_data = self.fetch_reddit(ticker)
            combined_data.extend(news_data + reddit_data)

        return combined_data
    
    def summarize_text(self,text):
        """
        Summarizes the text using Cohere's API.

        Args:
            text (str): The input text to summarize.

        Returns:
            str: The summarized text 
        """
        response = self.co.summarize(
              text=text,
              length="short",  
              format="paragraph",  
              model="summarize-xlarge" 
          )
        return response.summary 

    def process_combined_data_with_summary(self):
        """
        Processes the data fetched from the pipeline to include time, aggregated texts, and summaries.

        Returns:
            None: Updates self.agg_text with aggregated and summarized text for each ticker.
        """
        try:
            # Define EST timezone
            est = pytz.timezone("America/New_York")

            # Fetch combined data
            raw_data = self.fetch_combined_data()

            # Ensure raw_data is not empty
            if not raw_data:
                logging.warning("No data fetched from pipeline.")
                # Set default blank texts for all tickers
                for ticker in self.tickers:
                    self.agg_text[ticker] = "No data available."
                return

            # Add current timestamp in EST
            for entry in raw_data:
                entry["time"] = datetime.utcnow().replace(tzinfo=pytz.UTC).astimezone(est).strftime("%Y-%m-%d %H:%M:%S")

            # Convert raw data to a DataFrame
            df = pd.DataFrame(raw_data)

            # Ensure the DataFrame has the required columns
            if "ticker" not in df.columns or "text" not in df.columns:
                logging.error("Missing required columns in fetched data.")
                return

            # Aggregate texts by ticker
            aggregated_df = df.groupby("ticker").agg({
                "time": "first",  # Use the first timestamp
                "text": lambda texts: " ".join(texts)  # Concatenate all texts for each ticker
            }).reset_index()

            # Summarize aggregated texts
            aggregated_df["summary"] = aggregated_df["text"].apply(self.summarize_text)

            # Update self.agg_text with summarized data
            for ticker in self.tickers:
                filtered_df = aggregated_df[aggregated_df["ticker"] == ticker]
                if not filtered_df.empty:
                    self.agg_text[ticker] = filtered_df["summary"].values[0]
                else:
                    # Default to blank text if no data is available for this ticker
                    self.agg_text[ticker] = ""

            logging.info("Aggregated and summarized text data successfully.")
        
        except Exception as e:
            logging.error(f"Error in processing combined data with summary: {str(e)}")
            
    # Define function for sentiment analysis
    def get_sentiment(self, text):
        if not text:  # If the text is empty, return neutral sentiment
            return "Neutral", 1.0  # Neutral with logit probability 1.0
        len_text = min(len(text),2000)
        text = text[:len_text]
        # Define the prompt for the model
        prompt = f'''Instruction: What is the sentiment of this news? Please choose an answer from [Positive, Negative, Neutral].\nInput: {text}\nAnswer: '''

        # Tokenize directly on the GPU for efficiency
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, max_length=128).to(device)

        # Forward pass on GPU
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get logits for the last token and move them back to CPU
        logits = outputs.logits[:, -1, :].to("cpu")
        probs = torch.softmax(logits, dim=-1)

        # Class tokens for Positive, Negative, Neutral
        class_tokens = self.tokenizer(["Positive", "Negative", "Neutral"], add_special_tokens=False)["input_ids"]
        class_probs = {self.tokenizer.decode(token_id): probs[0, token_id].item() for token_id in class_tokens}

        # Get the most probable sentiment
        sentiment = max(class_probs, key=class_probs.get)

        # Clear intermediate variables
        del inputs, outputs, logits, probs
        torch.cuda.empty_cache()
        return sentiment, np.round(class_probs[sentiment],2)
        
    
    def run_periodically(self):
        """
        Periodically process combined data and update summaries at the start of every minute.
        """
        while True:
            # Wait until the next minute starts
            current_time = datetime.now()
            sleep_seconds = 60 - current_time.second
            time.sleep(sleep_seconds)

            try:
                # Process combined data with summaries
                self.process_combined_data_with_summary()
                logging.info(f"Text aggregation and summarization completed at {datetime.now()}")
                for ticker in self.tickers:
                    self.sentiment[ticker], self.prob[ticker] = self.get_sentiment(self.agg_text[ticker])
                    #Code for testinig 
                    # sentiments = ['Positive', 'Neutral', 'Negative']
                    # self.sentiment[ticker] = random.choice(sentiments)
                    # self.prob[ticker] = np.round(np.random.uniform(0.4, 0.5),2)
                    logging.info(f"Sentiment analysis finished for ticker: {ticker}")
            except Exception as e:
                logging.error(f"Error during periodic text aggregation: {str(e)}")
                
 
