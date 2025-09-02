from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
from datetime import datetime
from LiveStockPricePipeline import FinnhubWebSocket
from TextFetchPipeline import TextFetchPipeline
from SignalGenerator import SignalGeneration
import threading
from zoneinfo import ZoneInfo
from transformers import LlamaForCausalLM, LlamaTokenizerFast
from peft import PeftModel
import torch
from huggingface_hub import login
from dotenv import load_dotenv
import os

app = FastAPI()

# Load the .env file
load_dotenv()


# Load keys from .env file
reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET')
reddit_client_id = os.getenv('REDDIT_CLIENT_ID')
reddit_user_agent = os.getenv('REDDIT_USER_AGENT')
news_api_key = os.getenv('NEWS_API_KEY')
cohere_key = os.getenv('COHERE_KEY')
finnhub_token = os.getenv('FINNHUB_TOKEN')
hf_token = os.getenv('HF_TOKEN')

if not all([reddit_client_secret, reddit_client_id, reddit_user_agent, news_api_key, cohere_key, finnhub_token, hf_token]):
    raise EnvironmentError("Missing one or more environment variables. Please specify them in the .env file.")

# Global variables
tickers = ['AAPL', 'AMZN', 'TSLA']  # Initial tickers
pipeline_threads = []
trade_log = []  # Store trade logs

# Load model and tokenizer globally 
def load_model_and_tokenizer(hf_token):
    # Authenticate Hugging Face Hub
    login(hf_token)

    # Model details
    base_model = "meta-llama/Meta-Llama-3-8B"
    peft_model = "FinGPT/fingpt-mt_llama3-8b_lora"

    # Load tokenizer
    tokenizer = LlamaTokenizerFast.from_pretrained(base_model, trust_remote_code=True, use_auth_token=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load base model with 16-bit precision
    model = LlamaForCausalLM.from_pretrained(
        base_model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16
    )

    # Apply LoRA-based PEFT model
    model = PeftModel.from_pretrained(model, peft_model, torch_dtype=torch.float16)
    model = model.eval()

    return model, tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(hf_token)

# Initialize pipelines with pre-loaded model and tokenizer
text_pipeline = TextFetchPipeline(
    news_api_key,
    reddit_client_id,
    reddit_client_secret,
    reddit_user_agent,
    cohere_key, 
    tickers,
    model=model,
    tokenizer=tokenizer
)
stock_pipeline = FinnhubWebSocket(finnhub_token, tickers)
signal_generator = SignalGeneration(buffer_size=30)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("dynamic_tickers.log")]
)

# Stop and restart pipelines
def stop_pipelines():
    global stock_pipeline, text_pipeline, signal_generator, pipeline_threads
    if stock_pipeline:
        stock_pipeline.stop()
    pipeline_threads = []

def start_pipelines():
    global stock_pipeline, text_pipeline, signal_generator, pipeline_threads

    # Stop existing pipelines
    stop_pipelines()

    # Restart pipelines
    stock_pipeline = FinnhubWebSocket(finnhub_token, tickers)
    text_pipeline = TextFetchPipeline(
        news_api_key,
        reddit_client_id,
        reddit_client_secret,
        reddit_user_agent,
        cohere_key, 
        tickers, 
        model=model, 
        tokenizer=tokenizer
    )
    signal_generator = SignalGeneration(buffer_size=30)

    # Start WebSocket pipeline
    ws_thread = threading.Thread(target=stock_pipeline.start, daemon=True)
    ws_thread.start()
    pipeline_threads.append(ws_thread)

    # Start VWAP collection
    vwap_thread = threading.Thread(
        target=signal_generator.collect_vwap,
        args=(stock_pipeline,),
        daemon=True
    )
    vwap_thread.start()
    pipeline_threads.append(vwap_thread)

    # Start text aggregation
    aggregation_thread = threading.Thread(
        target=text_pipeline.run_periodically,
        daemon=True
    )
    aggregation_thread.start()
    pipeline_threads.append(aggregation_thread)

@app.on_event("startup")
def initialize_pipelines():
    start_pipelines()

# Models
class CombinedData(BaseModel):
    ticker: str
    VWAP: float
    time: str
    text: str
    sentiment: int
    probability: float
    MA_Crossover: int
    RSI: int
    Breakout: int
    Oscillator: int
    Signal: int

@app.post("/update_tickers/")
def update_tickers(new_tickers: List[str]):
    """
    Update the list of tickers and restart pipelines.
    """
    global tickers
    if not new_tickers or not isinstance(new_tickers, list):
        raise HTTPException(status_code=400, detail="Tickers must be a non-empty list of strings.")

    tickers = [ticker.upper().strip() for ticker in new_tickers]
    logging.info(f"Updated tickers: {tickers}")

    # Restart pipelines for the updated tickers
    start_pipelines()
    return {"message": "Tickers updated successfully.", "tickers": tickers}

@app.get("/mock_data", response_model=List[CombinedData])
def get_mock_data():
    """
    Fetch the latest VWAP values and trading signals dynamically.
    """
    if not tickers:
        return []

    # Fetch the latest VWAP values
    latest_vwap = stock_pipeline.latest_vwap if stock_pipeline else {}
    # Fetch the latest signals
    latest_signals = signal_generator.get_signals() if signal_generator else {}
    # Fetch the latest news
    latest_news = text_pipeline.agg_text if text_pipeline else {}
    # Current timestamp
    current_time = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
    
    sentiment_map = {"Positive": 1, "Negative": -1, "Neutral": 0}  # Map sentiment strings to integers

    # Generate data dynamically for each ticker
    
    # for TESTING purposes only
    # sentiments = ['Positive', 'Neutral', 'Negative']
    # random_sentiment = random.choice(sentiments)
    # random_prob = np.round(np.random.uniform(0.4, 0.5),2)
    
    mock_data = [
        CombinedData(
            ticker=ticker,
            VWAP=latest_vwap.get(ticker, 0.0),
            time=current_time,
            text=latest_news.get(ticker, "No data available."),
            sentiment=sentiment_map.get(text_pipeline.sentiment.get(ticker, 'Neutral'), 0),
            probability=text_pipeline.prob.get(ticker, 1.0),
            MA_Crossover=latest_signals.get(ticker, {}).get("SMA", 0),
            RSI=latest_signals.get(ticker, {}).get("RSI", 0),
            Breakout=latest_signals.get(ticker, {}).get("Breakout", 0),
            Oscillator=latest_signals.get(ticker, {}).get("Stochastic", 0),
            Signal=1
        )
        for ticker in tickers
    ]
    return mock_data

# Buy and Sell endpoints
@app.post("/buy")
async def buy(ticker: str = Form(...), amount: int = Form(...), price: float = Form(...)):
    """
    Handle buy transactions.
    """
    trade_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Bought {amount} shares of {ticker} @ ${price}"
    trade_log.append(trade_entry)
    logging.info(trade_entry)
    return RedirectResponse(url="/", status_code=303)

@app.post("/sell")
async def sell(ticker: str = Form(...), amount: int = Form(...), price: float = Form(...)):
    """
    Handle sell transactions.
    """
    trade_entry = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Sold {amount} shares of {ticker} @ ${price}"
    trade_log.append(trade_entry)
    logging.info(trade_entry)
    return RedirectResponse(url="/", status_code=303)

#Dashboard
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Stock Dashboard</title>
        <style>
            table {{
                width: 100%;
                border-collapse: collapse;
            }}
            th, td {{
                padding: 10px;
                text-align: center;
                border: 1px solid black;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
        <script>
            async function fetchData() {{
                const response = await fetch('/mock_data');
                const data = await response.json();
                const tableBody = document.getElementById('table-body');
                tableBody.innerHTML = '';
                data.forEach(item => {{
                    const row = `
                        <tr>
                            <td>${{item.ticker}}</td>
                            <td>${{item.VWAP}}</td>
                            <td>${{item.time}}</td>
                            <td>${{item.text}}</td>
                            <td>${{item.sentiment === 1 ? 'Positive' : (item.sentiment === -1 ? 'Negative' : 'Neutral')}}</td>
                            <td>${{item.probability}}</td>
                            <td>${{item.MA_Crossover === 1 ? '✅' : (item.MA_Crossover === -1 ? '❌' : 'Hold')}}</td>
                            <td>${{item.RSI === 1 ? '✅' : (item.RSI === -1 ? '❌' : 'Hold')}}</td>
                            <td>${{item.Breakout === 1 ? '✅' : (item.Breakout === -1 ? '❌' : 'Hold')}}</td>
                            <td>${{item.Oscillator === 1 ? '✅' : (item.Oscillator === -1 ? '❌' : 'Hold')}}</td>
                            <td>
                                <form action="/buy" method="post">
                                    <input type="hidden" name="ticker" value="${{item.ticker}}">
                                    <input type="hidden" name="price" value="${{item.VWAP}}">
                                    <input type="number" name="amount" min="1" required>
                                    <button type="submit">Buy</button>
                                </form>
                            </td>
                            <td>
                                <form action="/sell" method="post">
                                    <input type="hidden" name="ticker" value="${{item.ticker}}">
                                    <input type="hidden" name="price" value="${{item.VWAP}}">
                                    <input type="number" name="amount" min="1" required>
                                    <button type="submit">Sell</button>
                                </form>
                            </td>
                        </tr>`;
                    tableBody.insertAdjacentHTML('beforeend', row);
                }});
            }}

            async function updateTickers() {{
                const tickersInput = document.getElementById('tickers-input').value;
                const tickers = tickersInput.split(',').map(ticker => ticker.trim());
                await fetch('/update_tickers/', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json',
                    }},
                    body: JSON.stringify(tickers),
                }});
                alert('Tickers updated successfully!');
                fetchData();
            }}

            setInterval(fetchData, 5000);
            window.onload = fetchData;
        </script>
    </head>
    <body>
        <h1>Stock Dashboard</h1>
        <input type="text" id="tickers-input" placeholder="Enter tickers separated by commas">
        <button onclick="updateTickers()">Update Tickers</button>
        <table border="1">
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>VWAP</th>
                    <th>Time</th>
                    <th>Text</th>
                    <th>Sentiment</th>
                    <th>Probability</th>
                    <th>MA Crossover</th>
                    <th>RSI</th>
                    <th>Breakout</th>
                    <th>Oscillator</th>
                    <th>Buy</th>
                    <th>Sell</th>
                </tr>
            </thead>
            <tbody id="table-body"></tbody>
        </table>
        <h2>Trade Log</h2>
        <div>
            <ul>
                {"".join([f"<li>{escape(entry)}</li>" for entry in trade_log])}
            </ul>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
