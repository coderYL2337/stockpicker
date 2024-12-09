import streamlit as st
import pinecone
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import yfinance as yf
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from openai import OpenAI, Client
from groq import Groq, Client
from typing import Dict, List, Tuple
import os
from dotenv import load_dotenv
from xml.etree import ElementTree
import time
from bs4 import BeautifulSoup
import json

# Load environment variables
load_dotenv()

# Load the company_tickers.json file
def load_ticker_to_cik_mapping():
    try:
        file_path = os.path.join(os.getcwd(), "company_tickers.json")
        with open(file_path, "r") as f:
            data = json.load(f)
        
        # Create a dictionary mapping tickers to CIKs
        ticker_to_cik = {
            company_data["ticker"]: str(company_data["cik_str"]).zfill(10)
            for company_data in data.values()
        }
        return ticker_to_cik
    except Exception as e:
        print(f"Error loading ticker-to-CIK mapping: {e}")
        return {}
    
ticker_to_cik_mapping = load_ticker_to_cik_mapping()

# Initialize OpenAI client
# client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
client=OpenAI(api_key=os.getenv('GROQ_API_KEY'), base_url="https://api.groq.com/openai/v1")
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = os.getenv("PINECONE_INDEX")
index = pc.Index(index_name)

# Initialize sentence transformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# eodhdkey=os.getenv("EODHD_API_KEY")

def try_gemini_request(messages: List[Dict]) -> str:
    """Attempt to generate response with Gemini model"""
    try:
        # Convert chat messages to Gemini format
        system_message = next((msg['content'] for msg in messages if msg['role'] == 'system'), '')
        user_message = next((msg['content'] for msg in messages if msg['role'] == 'user'), '')
        
        # Combine system and user messages
        combined_message = f"{system_message}\n\nUser Query: {user_message}"
        
        # Generate response using Gemini
        model = genai.GenerativeModel('gemini-1.5-pro')
        response = model.generate_content(combined_message)
        
        return response.text
    except Exception as e:
        print(f"Failed to use Gemini: {str(e)}")
        return None


def try_llm_request(client, model_name: str, messages: List[Dict]) -> str:
    """Attempt to generate response with fallback to different models"""
    # Try primary model (Llama)
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Failed to use {model_name}: {str(e)}")
        
        # Try Gemini as first fallback
        gemini_response = try_gemini_request(messages)
        if gemini_response:
            return gemini_response
            
        # Try GPT-4 as second fallback
        try:
            openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Failed to use GPT-4: {str(e)}")
            return None
    
def enhance_search_query_with_llm(user_query: str) -> str:
    """Use LLM to create a detailed search query"""
    messages = [
        {"role": "system", "content": "You are a stock market expert. Convert the user's search query into a detailed description focusing on business models, industry terms, and key characteristics that would help find relevant stocks."},
        {"role": "user", "content": f"Create a detailed search query for: {user_query}"}
    ]
    
    #enhanced_query = try_llm_request(client, "gpt-4o-mini", messages)
    enhanced_query = try_llm_request(client, "llama-3.3-70b-versatile", messages)
    print(f"\nOriginal Query: {user_query}")
    print(f"Enhanced Query: {enhanced_query}")
    return enhanced_query or user_query

def search_stocks_in_pinecone(enhanced_query: str) -> List[Dict]:
    """Search Pinecone index using the enhanced query"""
    try:
        query_embedding = model.encode(enhanced_query).tolist()
        
        print("\nDebug Information:")
        print(f"Query Embedding (first 5 values): {query_embedding[:5]}")
        
        results = index.query(
            vector=query_embedding,
            namespace="stock-descriptions",
            top_k=6,
            include_metadata=True
        )
        
        print(f"\nNumber of matches found: {len(results.matches)}")
        if results.matches:
            for i, match in enumerate(results.matches):
                print(f"\nMatch {i+1}:")
                print(f"Score: {match.score}")
                print(f"Metadata keys: {match.metadata.keys()}")
                print(f"Business Summary: {match.metadata.get('Business Summary', '')[:100]}...")
        
        return results.matches
        
    except Exception as e:
        print(f"\nError during Pinecone search: {str(e)}")
        return []

def get_historical_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get historical data for a single ticker using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_date, end=end_date, interval='1d')  # Specify interval='1d'
        if df.empty:
            print(f"No historical data found for {ticker}")
            return pd.DataFrame()
        df.reset_index(inplace=True)  # Convert Date from index to column
        return df
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return pd.DataFrame()


def get_all_historical_data(tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get and combine historical data for multiple tickers"""
    all_data = []
    
    for ticker in tickers:
        hist_data = get_historical_data(ticker, start_date, end_date)
        if not hist_data.empty:
            hist_data['ticker'] = ticker  # Add ticker column
            all_data.append(hist_data)
    
    if not all_data:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_data, ignore_index=True)
    # Make sure date is datetime
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    
    return combined_df


def get_yfinance_data(ticker: str) -> Dict:
    """Get stock data from yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'website': info.get('website', ''),
            'earningsGrowth': info.get('earningsGrowth', 0) * 100 if isinstance(info.get('earningsGrowth'), (int, float)) else 0,
            'revenueGrowth': info.get('revenueGrowth', 0) * 100 if isinstance(info.get('revenueGrowth'), (int, float)) else 0,
            'grossMargins': info.get('grossMargins', 0) * 100 if isinstance(info.get('grossMargins'), (int, float)) else 0,
            'ebitdaMargins': info.get('ebitdaMargins', 0) * 100 if isinstance(info.get('ebitdaMargins'), (int, float)) else 0,
            '52WeekChange': info.get('52WeekChange', 0),
            'trailingPE': info.get('trailingPE', 0),
            'beta': info.get('beta', 0),
            'marketCap': info.get('marketCap', 0)
        }
    except Exception as e:
        print(f"Error fetching yfinance data for {ticker}: {str(e)}")
        return {}

def create_price_comparison_chart(hist_data: pd.DataFrame, selected_tickers: List[str] = None) -> None:
    """Create normalized price comparison chart with optional ticker selection"""
    if hist_data.empty:
        st.warning("No historical data available for comparison")
        return
        
    # Only use selected tickers if provided
    if selected_tickers:
        price_data = hist_data.pivot(index='Date', columns='ticker', values='Close')
        price_data = price_data[selected_tickers]  # Filter for selected tickers only
    else:
        price_data = hist_data.pivot(index='date', columns='ticker', values='close')
    
    normalized_data = price_data.apply(lambda x: (x / x.iloc[0] - 1) * 100)
    
    fig = px.line(normalized_data,
                  title='Normalized Stock Price History (% Change)',
                  labels={'value': 'Price Change (%)',
                         'date': 'Date',
                         'variable': 'Stock'})
    
    fig.update_layout(
        legend_title_text='Stocks',
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def generate_analysis_with_llm(stocks_data: List[dict]) -> str:
    """Generate analysis text using LLM"""
    messages = [
        {"role": "system", "content": "You are a stock market analyst. Provide a concise analysis of the following stocks, comparing their performance metrics and business models. Focus on key differentiators and potential opportunities/risks."},
        {"role": "user", "content": f"Analyze these stocks and their metrics: {stocks_data}"}
    ]
    
    # analysis = try_llm_request(client, "gpt-4o-mini", messages)
    analysis = try_llm_request(client, "llama-3.3-70b-versatile", messages)
    return analysis or "Unable to generate analysis at this time."

# Helper functions for formatting
def format_market_cap(value):
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    else:
        return f"${value:,.0f}"

def format_number(value):
    return f"{value:,}"

def calculate_stock_metrics(stock, stock_metadata):
    """Calculate metrics for each stock"""
    metrics = {
        'Financial Health': (
            (stock['yfinance_data'].get('grossMargins', 0) + 
             stock['yfinance_data'].get('ebitdaMargins', 0)) / 2) * 10,
            
        'Market Competition': min(stock['yfinance_data'].get('marketCap', 0) / 1e11, 10),
        
        'Growth Potential': min(
            ((stock['yfinance_data'].get('earningsGrowth', 0) + 
              stock['yfinance_data'].get('revenueGrowth', 0)) / 2) * 2, 10),
              
        'Innovation': min(5 + stock['yfinance_data'].get('earningsGrowth', 0) / 10, 10),
        
        'Industry Trends': min((stock['yfinance_data'].get('52WeekChange', 0) * 10 + 5), 10),
        
        'Regulatory Environment': 7  # Base score, could be adjusted based on sector
    }
    return metrics

def generate_factor_explanation(stock, metrics):
    """Generate explanations for each factor"""
    return {
        'Growth Potential': f"{'Strong' if metrics['Growth Potential'] > 7 else 'Moderate'} earnings growth suggests {'significant' if metrics['Growth Potential'] > 7 else 'steady'} future market expansion.",
        'Market Competition': f"{'Leading' if metrics['Market Competition'] > 7 else 'Competitive'} market position with {'strong presence' if metrics['Market Competition'] > 7 else 'growing influence'}.",
        'Financial Health': f"{'Strong' if metrics['Financial Health'] > 7 else 'Solid'} gross and EBITDA margins indicate {'exceptional' if metrics['Financial Health'] > 7 else 'stable'} financial health.",
        'Innovation': f"{'Leading innovator' if metrics['Innovation'] > 7 else 'Active participant'} in industry developments and technological advancement.",
        'Industry Trends': f"{'Strong alignment' if metrics['Industry Trends'] > 7 else 'Good alignment'} with current industry trends and market dynamics.",
        'Regulatory Environment': f"{'Favorable' if metrics['Regulatory Environment'] > 7 else 'Manageable'} regulatory landscape for operations and growth."
    }

def add_market_trend_radar(all_stock_data):
    """Add Market Trend Radar section to the app"""
    st.subheader("Market Trend Analysis")

        # Define distinct colors
    colors = {
        0: 'rgb(255, 0, 0)',      # Red
        1: 'rgb(0, 0, 255)',      # Blue
        2: 'rgb(0, 128, 0)',      # Green
        3: 'rgb(128, 0, 128)',    # Purple
        4: 'rgb(255, 165, 0)',    # Orange
        5: 'rgb(165, 42, 42)'     # Brown
    }

    # Create metrics for each stock
    stock_metrics = {}
    for stock in all_stock_data:
        stock_metrics[stock['ticker']] = calculate_stock_metrics(stock, st.session_state.stock_metadata)

    # Create radar chart
    factors = ['Financial Health', 'Market Competition', 'Growth Potential', 
              'Innovation', 'Industry Trends', 'Regulatory Environment']

    # Let user select stocks to compare
    selected_tickers = st.multiselect(
        "Select stocks to compare:",
        options=[stock['ticker'] for stock in all_stock_data],
        default=[stock['ticker'] for stock in all_stock_data]
    )

    if selected_tickers:
        # Create radar chart
        fig = go.Figure()
        
        for idx,ticker in enumerate(selected_tickers):
            metrics = stock_metrics[ticker]
            fig.add_trace(go.Scatterpolar(
                r=[metrics[factor] for factor in factors],
                theta=factors,
                fill='toself',
                name=ticker,
                line_color=colors[idx],          # Set line color
                fillcolor=colors[idx],           # Set fill color
                opacity=0.3                      # Make fill slightly transparent
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )),
            showlegend=True,
            title="Market Trend Radar - All Stocks"
        )

        st.plotly_chart(fig, use_container_width=True)

        # Show factor explanations
        st.subheader("Factor Explanations")
        for ticker in selected_tickers:
            stock = next(s for s in all_stock_data if s['ticker'] == ticker)
            metrics = stock_metrics[ticker]
            explanations = generate_factor_explanation(stock, metrics)
            
            with st.expander(f"Explanations for {ticker}"):
                for factor, explanation in explanations.items():
                    st.write(f"**{factor}:** {explanation}")

def fetch_stock_news(ticker: str, max_articles: int = 5, days_back: int = 7) -> List[Dict]:
    """
    Fetch news for a given stock ticker from Yahoo Finance RSS
    
    Args:
        ticker (str): Stock ticker symbol
        max_articles (int): Maximum number of articles to fetch (default: 5)
        days_back (int): Only fetch articles from the last N days (default: 7)
    """
    rss_feed_url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(rss_feed_url, headers=headers)
        if response.status_code == 200:
            root = ElementTree.fromstring(response.content)
            news_items = []
            all_news_items = []  # Store all news items for fallback
            current_time = datetime.now()
            
            # First pass: collect all articles and their dates
            for item in root.findall('./channel/item'):
                pub_date = datetime.strptime(item.find('pubDate').text, '%a, %d %b %Y %H:%M:%S %z')
                days_old = (current_time - pub_date.replace(tzinfo=None)).days
                
                news_item = {
                    'title': item.find('title').text,
                    'link': item.find('link').text,
                    'pubDate': item.find('pubDate').text,
                    'description': item.find('description').text,
                    'ticker': ticker,
                    'days_ago': days_old
                }
                
                # Add to the main list if within days_back
                if days_old <= days_back:
                    news_items.append(news_item)
                
                # Add to all_news_items for potential fallback
                all_news_items.append(news_item)
                
                # Break if we've reached max_articles for recent news
                if len(news_items) >= max_articles:
                    break
            
            # If we don't have enough recent articles, use the most recent available
            if len(news_items) < max_articles:
                print(f"Warning: Only found {len(news_items)} recent articles for {ticker}. Using older articles as fallback.")
                # Sort all_news_items by date (most recent first)
                all_news_items.sort(key=lambda x: x['days_ago'])
                
                # Take the most recent articles up to max_articles
                news_items = all_news_items[:max_articles]
            
            return news_items
            
    except Exception as e:
        print(f"Error fetching news for {ticker}: {str(e)}")
        return []



def analyze_single_article(article: Dict) -> Tuple[float, Dict]:
    """Analyze sentiment of a single news article"""
    messages = [
        {"role": "system", "content": """You are a financial analyst. Analyze this news article and provide:
        1. A numerical sentiment score from 0 to 100 (just the number)
        2. A brief summary of key developments
        3. Key risks and challenges identified
        4. Future outlook implications

        Format your response exactly like this:
        SENTIMENT: [score]
        DEVELOPMENTS: [summary]
        RISKS: [risks]
        OUTLOOK: [outlook]"""},
        {"role": "user", "content": f"Analyze this news article: Title: {article['title']}\nContent: {article['description']}"}
    ]
    
    response = try_llm_request(client, "llama-3.3-70b-versatile", messages)
    
    if not response:
        return 50.0, {
            "recent_developments": "Analysis unavailable",
            "risks_challenges": "Analysis unavailable",
            "future_outlook": "Analysis unavailable"
        }
    
    try:
        # Parse the response line by line
        lines = response.split('\n')
        sentiment_score = 50.0  # Default score
        developments = "No developments available"
        risks = "No risks available"
        outlook = "No outlook available"
        
        for line in lines:
            line = line.strip()
            if line.startswith('SENTIMENT:'):
                try:
                    score = line.split('SENTIMENT:')[1].strip()
                    score = ''.join(c for c in score if c.isdigit() or c == '.')
                    sentiment_score = float(score)
                except:
                    sentiment_score = 50.0
            elif line.startswith('DEVELOPMENTS:'):
                developments = line.split('DEVELOPMENTS:')[1].strip()
            elif line.startswith('RISKS:'):
                risks = line.split('RISKS:')[1].strip()
            elif line.startswith('OUTLOOK:'):
                outlook = line.split('OUTLOOK:')[1].strip()
        
        return sentiment_score, {
            "recent_developments": developments,
            "risks_challenges": risks,
            "future_outlook": outlook
        }
        
    except Exception as e:
        print(f"Error parsing sentiment analysis: {str(e)}")
        return 50.0, {
            "recent_developments": "Error analyzing developments",
            "risks_challenges": "Error analyzing risks",
            "future_outlook": "Error analyzing outlook"
        }

def analyze_news_sentiment(news_items: List[Dict]) -> Tuple[float, Dict]:
    """Analyze sentiment of all news items for a stock"""
    if not news_items:
        return 50.0, {
            "recent_developments": "No news available",
            "risks_challenges": "No news available",
            "future_outlook": "No news available"
        }
    
    article_sentiments = []
    latest_analysis = None
    
    # Analyze each article individually
    for item in news_items:
        sentiment_score, analysis = analyze_single_article(item)
        article_sentiments.append(sentiment_score)
        item['sentiment_score'] = sentiment_score  # Add sentiment score to the item
        item.update(analysis)  # Add analysis details to the item
        if latest_analysis is None:
            latest_analysis = analysis
    
    # Calculate average sentiment
    avg_sentiment = sum(article_sentiments) / len(article_sentiments)
    
    return avg_sentiment, latest_analysis


def add_sentiment_analysis(all_stock_data: List[Dict]):
    """Add sentiment analysis section to the app"""
    st.subheader("Sentiment Analysis")
    
    # Fetch and analyze news for each stock
    sentiment_data = []
    news_data = []
    
    with st.spinner("Analyzing recent news..."):
        for stock in all_stock_data:
            ticker = stock['ticker']
            news_items = fetch_stock_news(ticker, max_articles=5, days_back=7)
            sentiment_score, analysis = analyze_news_sentiment(news_items)
            
            sentiment_data.append({
                'ticker': ticker,
                'sentiment_score': sentiment_score,
                'name': stock['name']
            })
            
            # Store news items with analysis
            for item in news_items:
                item.update(analysis)
                news_data.append(item)
    
    # Create sentiment score chart with custom color scheme
    sentiment_df = pd.DataFrame(sentiment_data)
    sentiment_df = sentiment_df.sort_values('sentiment_score', ascending=False)
    
    # Use a custom color scale that ensures visibility
    fig = px.bar(sentiment_df,
                 x='ticker',
                 y='sentiment_score',
                 title='Average Sentiment Score by Stock',
                 labels={'ticker': 'Stock Symbol', 'sentiment_score': 'Average Sentiment Score'})
    
    # Update the color scheme to use solid, easily distinguishable colors
    fig.update_traces(marker_color=[
        'rgb(25, 25, 112)',  # Dark Blue
        'rgb(0, 71, 171)',   # Medium Blue
        'rgb(30, 144, 255)', # Dodger Blue
        'rgb(0, 119, 190)',  # Ocean Blue
        'rgb(0, 147, 175)',  # Blue Green
        'rgb(0, 163, 204)'   # Light Blue
    ])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display rankings and analysis
    st.subheader("Stock Rankings based on News Sentiment:")
    for _, row in sentiment_df.iterrows():
        stock_news = [item for item in news_data if item['ticker'] == row['ticker']]
        if stock_news:
            analysis = stock_news[0]
            st.write(f"{row['ticker']} ({row['name']}): Sentiment score {row['sentiment_score']:.2f}")
            with st.expander(f"Details for {row['ticker']}"):
                st.write("**Recent Developments:**", analysis.get('recent_developments', 'N/A'))
                st.write("**Risks & Challenges:**", analysis.get('risks_challenges', 'N/A'))
                st.write("**Future Outlook:**", analysis.get('future_outlook', 'N/A'))
    
    # Display detailed news data in a scrollable window
    st.subheader("Detailed News Data")
    
    # Create a DataFrame with all the news data
    detailed_news_df = pd.DataFrame([
        {
            'URL': item['link'],
            'Published Time': item['pubDate'],
            'Related Tickers': item['ticker'],
            'Scraped Text': item.get('description', ''),
            'Sentiment': item.get('sentiment_score', ''),
            'Recent Developments': item.get('recent_developments', ''),
            'Risks & Challenges': item.get('risks_challenges', ''),
            'Future Outlook': item.get('future_outlook', '')
        }
        for item in news_data
    ])
    
    # Display the DataFrame in a scrollable container
    st.dataframe(
        detailed_news_df,
        use_container_width=True,
        height=300  # Adjust height as needed
    )
    
    # Create downloadable CSV
    csv = detailed_news_df.to_csv(index=False)
    st.download_button(
        label="Download News Data as CSV",
        data=csv,
        file_name="stock_news_analysis.csv",
        mime="text/csv"
    )


def get_latest_10q_filing(ticker: str) -> Tuple[str, str, str]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; CompanyResearch/1.0; yourname@email.com)'
        }
        
        # Step 1: Retrieve CIK from local mapping
        cik = ticker_to_cik_mapping.get(ticker.upper())
        if not cik:
            print(f"No CIK found for {ticker} in local mapping.")
            return None, None, None

        print(f"Ticker: {ticker}, CIK: {cik}")

        # Step 2: Fetch filings from SEC submissions API
        filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        response = requests.get(filings_url, headers=headers)
        if response.status_code != 200:
            print(f"Failed to fetch filings data for {ticker}. Response: {response.text}")
            return None, None, None

        filings_data = response.json()
        recent_filings = filings_data.get("filings", {}).get("recent", {})
        
        # Step 3: Find the latest 10-Q filing
        for idx, form in enumerate(recent_filings.get("form", [])):
            if form == "10-Q":
                filing_date = recent_filings["filingDate"][idx]
                accession_number = recent_filings["accessionNumber"][idx].replace('-', '')
                document_url = recent_filings["primaryDocument"][idx]
                filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_number}/{document_url}"
                return filing_date, filing_url, "Filing text not fetched yet"
        
        print(f"No 10-Q filing found for {ticker}")
        return None, None, None
    except Exception as e:
        print(f"Error fetching 10-Q for {ticker}: {str(e)}")
        return None, None, None


def analyze_10q_with_llm(filing_text: str) -> Dict:
    """Analyze 10-Q filing text using LLM with specific financial metrics focus"""
    messages = [
        {"role": "system", "content": """You are a financial analyst. Analyze this 10-Q filing and provide detailed scores 
        (0-100) for the following factors, focusing on specific financial metrics and indicators:
        
        1. Performance: Analyze revenue growth, profit margins, EPS trends
        2. Growth Potential: Evaluate R&D investment, market expansion plans, new product developments
        3. Risk: Assess debt levels, regulatory challenges, market competition
        4. Competitive Edge: Evaluate market share, patents/IP, brand strength
        
        Base your scoring on concrete financial metrics and provide specific justification.
        
        Format your response exactly like this:
        PERFORMANCE: [score]
        PERFORMANCE_DETAILS: [specific metrics and reasons]
        GROWTH_POTENTIAL: [score]
        GROWTH_DETAILS: [specific metrics and reasons]
        RISK: [score]
        RISK_DETAILS: [specific metrics and reasons]
        COMPETITIVE_EDGE: [score]
        COMPETITIVE_DETAILS: [specific metrics and reasons]"""},
        {"role": "user", "content": f"Analyze this 10-Q filing data: {filing_text[:8000]}"}  # Increased context
    ]
    
    response = try_llm_request(client, "llama-3.3-70b-versatile", messages)
    
    if not response:
        return {
            "performance": 0,
            "growth_potential": 0,
            "risk": 0,
            "competitive_edge": 0
        }
    
    try:
        scores = {
            "performance": 0,
            "growth_potential": 0,
            "risk": 0,
            "competitive_edge": 0
        }
        
        # Parse scores and details
        lines = response.split('\n')
        for line in lines:
            if ':' not in line:
                continue
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if key == 'performance':
                scores['performance'] = int(float(value))
            elif key == 'growth_potential':
                scores['growth_potential'] = int(float(value))
            elif key == 'risk':
                scores['risk'] = int(float(value))
            elif key == 'competitive_edge':
                scores['competitive_edge'] = int(float(value))
                
        return scores
        
    except Exception as e:
        print(f"Error parsing LLM response: {str(e)}")
        return {
            "performance": 0,
            "growth_potential": 0,
            "risk": 0,
            "competitive_edge": 0
        }


def create_progress_bar(value: int, color: str) -> str:
    """Create HTML for a progress bar"""
    return f"""
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background-color: #1E1E1E; width: 200px; height: 20px; border-radius: 10px; overflow: hidden;">
                <div style="width: {value}%; height: 100%; background-color: {color}; transition: width 0.5s;"></div>
            </div>
            <span style="color: {color};">{value}</span>
        </div>
    """
def add_sec_filings_section(all_stock_data: List[Dict]):
    """Add SEC Filings section to the app"""
    st.subheader("SEC Filings")
    
    for stock in all_stock_data:
        ticker = stock['ticker']
        
        try:
            filing_date, filing_url, filing_text = get_latest_10q_filing(ticker)
            
            # Create two columns for layout
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{ticker}**")
                if filing_date and filing_url and filing_text:
                    st.write("Periodic Financial Reports")
                    st.write(filing_date)
                    st.markdown(f"[View 10-Q Filing]({filing_url})")
                else:
                    st.write("N/A")
                    st.write("No filing available")
            
            with col2:
                st.write("Key Insights:")
                if filing_text:
                    insights = analyze_10q_with_llm(filing_text)
                    
                    # Create progress bars with different colors
                    st.markdown(f"""
                        Performance: {create_progress_bar(insights['performance'], '#4CAF50')}
                        Growth Potential: {create_progress_bar(insights['growth_potential'], '#2196F3')}
                        Risk: {create_progress_bar(insights['risk'], '#f44336')}
                        Competitive Edge: {create_progress_bar(insights['competitive_edge'], '#9C27B0')}
                    """, unsafe_allow_html=True)
                else:
                    st.write("No insights available")
            
            st.markdown("---")  # Add separator between stocks
            
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            st.write(f"**{ticker}**")
            st.write("Error retrieving filing information")
            st.markdown("---")


def main():
    st.set_page_config(page_title="Stock Picker", layout="wide", page_icon="📈")
    
    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'all_stock_data' not in st.session_state:
        st.session_state.all_stock_data = None
    if 'stock_metadata' not in st.session_state:
        st.session_state.stock_metadata = None
    
    # Initial View
    st.title("Stock Picker 📈")
    st.write("Enter a description of the kinds of stocks you are looking for:")
    
    query = st.text_input(
        label="Stock Search",
        placeholder="e.g., 'electric car manufacturers' or 'cloud computing companies'",
        label_visibility="collapsed"
    )
    
    search_clicked = st.button("Find Stocks")



    def display_stock_data(stock_data_list):
        st.subheader("Found Stocks")
        cols = st.columns(2)

        for idx, stock in enumerate(stock_data_list):
            with cols[idx % 2]:
                with st.expander(stock['ticker'], expanded=True):
                    # Ticker and Name
                    st.write(f"## {stock['ticker']}")
                    st.write(f"_{stock['name']}_")

                    # Business Summary with expanded view handling
                    st.write("**Business Summary**")
                    business_summary = stock['description']
                    if len(business_summary) > 200:
                        button_id = f"expand_summary_{stock['ticker']}"
                        if button_id not in st.session_state:
                            st.session_state[button_id] = False
                            
                        if st.session_state[button_id]:
                            st.write(business_summary)
                            if st.button("Show Less ▲", key=f"collapse_{stock['ticker']}"):
                                st.session_state[button_id] = False
                                st.rerun()
                        else:
                            st.write(f"{business_summary[:200]}...")
                            if st.button("Show More ▼", key=f"expand_{stock['ticker']}"):
                                st.session_state[button_id] = True
                                st.rerun()
                    else:
                        st.write(business_summary)

                    # Website
                    st.write("**Website**")
                    st.write(stock['yfinance_data'].get('website', 'N/A'))

                    # Metrics in two columns
                    metrics_cols = st.columns(2)
                    
                    # Left column metrics
                    with metrics_cols[0]:
                        metrics_left = {
                            "EOD Price": f"${stock['realtime_data'].get('close', 0):.2f}",
                            "Market Cap": format_market_cap(stock['yfinance_data'].get('marketCap', 0)),
                            "Sector": st.session_state.stock_metadata[stock['ticker']].get('Sector', 'N/A'),
                            "P/E Ratio": f"{stock['yfinance_data'].get('trailingPE', 0):.2f}",
                            "Beta": f"{stock['yfinance_data'].get('beta', 0):.2f}",
                            "Earnings Growth": f"{stock['yfinance_data'].get('earningsGrowth', 0):.1f}%"
                        }
                        
                        for label, value in metrics_left.items():
                            st.write(f"**{label}**")
                            st.write(value)

                    # Right column metrics
                    with metrics_cols[1]:
                        # Add Yahoo Finance link first
                        yahoo_finance_url = f"https://finance.yahoo.com/quote/{stock['ticker']}"
                        st.write("**Live Quote**")
                        st.write(f"[View on Yahoo ↗]({yahoo_finance_url})")
                        
                        metrics_right = {
                            "Volume": format_number(stock['realtime_data'].get('volume', 0)),
                            "Revenue Growth": f"{stock['yfinance_data'].get('revenueGrowth', 0):.1f}%",
                            "Gross Margins": f"{stock['yfinance_data'].get('grossMargins', 0):.1f}%",
                            "EBITDA Margins": f"{stock['yfinance_data'].get('ebitdaMargins', 0):.1f}%",
                            "52 Week Change": f"{stock['yfinance_data'].get('52WeekChange', 0) * 100:.1f}%"
                        }
                        
                        for label, value in metrics_right.items():
                            st.write(f"**{label}**")
                            st.write(value)
    # Process new search
    if search_clicked and query:
        with st.spinner("Processing your request..."):
            # Get enhanced query and print debug info to console
            enhanced_query = enhance_search_query_with_llm(query)
            print(f"\nOriginal Query: {query}")
            print(f"Enhanced Query: {enhanced_query}")
            
            # Get search results
            search_results = search_stocks_in_pinecone(enhanced_query)
            if not search_results:
                st.error("No matching stocks found. Please try a different search query.")
                return

            # Extract tickers and create a mapping of ticker to metadata
            stock_metadata = {}
            tickers = []
            for result in search_results:
                if 'Ticker' in result.metadata:
                    ticker = result.metadata['Ticker']
                    tickers.append(ticker)
                    stock_metadata[ticker] = result.metadata
            
            print("\nFound Tickers:", tickers)


            try:
                all_stock_data = []
                for ticker in tickers:
                    metadata = stock_metadata[ticker]
                    yf_data = get_yfinance_data(ticker)
                    # rt_data = next((item for item in realtime_data if item['code'].replace('.US', '') == ticker), {})
                    # Get current price using valid period
                    try:
                        stock = yf.Ticker(ticker)
                        current_data = stock.history(period='1d')
                        current_price = current_data['Close'].iloc[-1] if not current_data.empty else 0
                        current_volume = current_data['Volume'].iloc[-1] if not current_data.empty else 0
                    except Exception as e:
                        print(f"Error fetching current price for {ticker}: {e}")
                        current_price = 0
                        current_volume = 0
                    stock_data = {
                        'ticker': ticker,
                        'name': metadata.get('Name', ''),
                        'description': metadata.get('Business Summary', ''),
                        'yfinance_data': yf_data,
                        'realtime_data': {
                            'close': current_price,
                            'volume': current_volume
                        }
                    }
                    all_stock_data.append(stock_data)
                
                # Save to session state
                st.session_state.search_results = search_results
                st.session_state.all_stock_data = all_stock_data
                st.session_state.stock_metadata = stock_metadata
                
                # Display stock data
                display_stock_data(all_stock_data)

                # Add download button for stock information
                if st.session_state.all_stock_data:
                    # Prepare data for CSV
                    stock_info_data = []
                    for stock in st.session_state.all_stock_data:
                        stock_info = {
                            'Ticker': stock['ticker'],
                            'Name': stock['name'],
                            'Business Summary': stock['description'],
                            'Website': stock['yfinance_data'].get('website', ''),
                            'EOD Price': stock['realtime_data'].get('close', 0),
                            'Volume': stock['realtime_data'].get('volume', 0),
                            'Market Cap': stock['yfinance_data'].get('marketCap', 0),
                            'Sector': stock_metadata[stock['ticker']].get('Sector', ''),
                            'P/E Ratio': stock['yfinance_data'].get('trailingPE', 0),
                            'Beta': stock['yfinance_data'].get('beta', 0),
                            'Earnings Growth': stock['yfinance_data'].get('earningsGrowth', 0),
                            'Revenue Growth': stock['yfinance_data'].get('revenueGrowth', 0),
                            'Gross Margins': stock['yfinance_data'].get('grossMargins', 0),
                            'EBITDA Margins': stock['yfinance_data'].get('ebitdaMargins', 0),
                            '52 Week Change': stock['yfinance_data'].get('52WeekChange', 0)
                        }
                        stock_info_data.append(stock_info)
                    
                    # Convert to DataFrame and then to CSV
                    info_df = pd.DataFrame(stock_info_data)
                    csv_data = info_df.to_csv(index=False)

                    
                    if 'stock_info_csv' not in st.session_state:
                        st.session_state.stock_info_csv = csv_data
                    
                    # Add download button
                    st.download_button(
                        label="Download Stock Information as CSV",
                        data=st.session_state.stock_info_csv,
                        file_name="stock_information.csv",
                        mime="text/csv",
                        key="download_info_button",
                        help="Download stock information for selected stocks"
                    )
                
                # Price comparison section
                st.subheader("Stock Price Comparison")
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start Date",
                        value=datetime.now() - timedelta(days=365),
                        key="start_date"
                    )
                with col2:
                    end_date = st.date_input(
                        "End Date",
                        value=datetime.now(),
                        key="end_date"
                    )

                # Add multiselect for tickers
                selected_tickers = st.multiselect(
                    "Select stocks to plot",
                    options=tickers,
                    default=tickers,
                    key="ticker_select"
                )
                
                if start_date and end_date:
                    historical_data = get_all_historical_data(selected_tickers, start_date, end_date)
                    create_price_comparison_chart(historical_data, selected_tickers)
                    
                    analysis = generate_analysis_with_llm(st.session_state.all_stock_data)
                    st.subheader("Stock Comparison Summary")
                    st.write(analysis)
                    
                    if not historical_data.empty:
                        csv = historical_data.to_csv(index=False)
                        if 'historical_csv' not in st.session_state:
                            st.session_state.historical_csv = csv
                        st.download_button(
                            label="Download Historical Data as CSV",
                            data=st.session_state.historical_csv,
                            file_name="historical_stock_data.csv",
                            mime="text/csv",
                            key="download_historical_button",
                            help="Download historical price data for selected stocks"
                        )
                        
            except Exception as e:
                st.error("Error fetching stock data")
                print("**Error Details:**")           # Level 4
                print(f"Error Type: {type(e)}...")    # Level 4
                st.error("Error fetching stock data")
        # After display_stock_data(all_stock_data)
        if 'all_stock_data' in locals():
            add_market_trend_radar(all_stock_data)
            add_sentiment_analysis(all_stock_data) 
            add_sec_filings_section(all_stock_data) 
    # Display from session state if we have previous results
    elif st.session_state.all_stock_data is not None:
        display_stock_data(st.session_state.all_stock_data)

         # Add download button for stock information
        if st.session_state.all_stock_data:
            # Prepare data for CSV
            stock_info_data = []
            for stock in st.session_state.all_stock_data:
                stock_info = {
                    'Ticker': stock['ticker'],
                    'Name': stock['name'],
                    'Business Summary': stock['description'],
                    'Website': stock['yfinance_data'].get('website', ''),
                    'EOD Price': stock['realtime_data'].get('close', 0),
                    'Volume': stock['realtime_data'].get('volume', 0),
                    'Market Cap': stock['yfinance_data'].get('marketCap', 0),
                    'Sector': st.session_state.stock_metadata[stock['ticker']].get('Sector', ''),
                    'P/E Ratio': stock['yfinance_data'].get('trailingPE', 0),
                    'Beta': stock['yfinance_data'].get('beta', 0),
                    'Earnings Growth': stock['yfinance_data'].get('earningsGrowth', 0),
                    'Revenue Growth': stock['yfinance_data'].get('revenueGrowth', 0),
                    'Gross Margins': stock['yfinance_data'].get('grossMargins', 0),
                    'EBITDA Margins': stock['yfinance_data'].get('ebitdaMargins', 0),
                    '52 Week Change': stock['yfinance_data'].get('52WeekChange', 0)
                }
                stock_info_data.append(stock_info)
            
            # Convert to DataFrame and then to CSV
            info_df = pd.DataFrame(stock_info_data)
            csv_data = info_df.to_csv(index=False)
            
            if 'stock_info_csv' not in st.session_state:
                st.session_state.stock_info_csv = csv_data
            
            # Add download button
            st.download_button(
                label="Download Stock Information as CSV",
                data=st.session_state.stock_info_csv,
                file_name="stock_information.csv",
                mime="text/csv",
                key="download_info_button",
                help="Download stock information for selected stocks"
            )
        
        # Recreate comparison section
       # st.subheader("Stock Price Comparison")
        st.subheader("Stock Price Comparison")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365),
                key="start_date"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.now(),
                key="end_date"
            )
        
        tickers = [stock['ticker'] for stock in st.session_state.all_stock_data]
        
        # Add multiselect for tickers
        selected_tickers = st.multiselect(
            "Select stocks to plot",
            options=tickers,
            default=tickers,
            key="ticker_select"
        )
        
        if start_date and end_date:
            historical_data = get_all_historical_data(selected_tickers, start_date, end_date)
            create_price_comparison_chart(historical_data, selected_tickers)
            
            # Re-generate analysis text
            analysis = generate_analysis_with_llm(st.session_state.all_stock_data)
            st.subheader("Stock Comparison Summary")
            st.write(analysis)
            
            if not historical_data.empty:
                csv = historical_data.to_csv(index=False)
                st.download_button(
                    label="Download Historical Data as CSV",
                    data=csv,
                    file_name="historical_stock_data.csv",
                    mime="text/csv",
                    key="download_historical_button",
                    help="Download historical price data for selected stocks"       
                )
        # Add market trend radar using session state data
        add_market_trend_radar(st.session_state.all_stock_data)
        add_sentiment_analysis(st.session_state.all_stock_data)
        add_sec_filings_section(st.session_state.all_stock_data) 
if __name__ == "__main__":
    main()
