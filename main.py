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
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index_name = os.getenv("PINECONE_INDEX")
index = pc.Index(index_name)

# Initialize sentence transformer
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# eodhdkey=os.getenv("EODHD_API_KEY")

def try_llm_request(client, model_name: str, messages: List[Dict]) -> str:
    """Attempt to generate response with a specific LLM model"""
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Failed to use {model_name}: {str(e)}")
        return None
def enhance_search_query_with_llm(user_query: str) -> str:
    """Use LLM to create a detailed search query"""
    messages = [
        {"role": "system", "content": "You are a stock market expert. Convert the user's search query into a detailed description focusing on business models, industry terms, and key characteristics that would help find relevant stocks."},
        {"role": "user", "content": f"Create a detailed search query for: {user_query}"}
    ]
    
    enhanced_query = try_llm_request(client, "gpt-4o-mini", messages)
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

def get_batch_realtime_data(tickers: List[str]) -> List[Dict]:
    """Get real-time data for multiple stocks in one API call"""
    if not tickers:
        return []
    
    main_ticker = tickers[0]
    other_tickers = ','.join(tickers[1:])
    
    eodhdkey = os.getenv("EODHD_API_KEY")
    url = f"https://eodhd.com/api/real-time/{main_ticker}?s={other_tickers}&api_token={eodhdkey}&fmt=json"
    
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        print(f"Error fetching batch real-time data: {e}")
        return []

def get_historical_data(ticker: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    """Get historical data for a single ticker within date range"""
    eodhdkey = os.getenv("EODHD_API_KEY")
    
    url = f"https://eodhd.com/api/table.csv?s={ticker}&a={start_date.month}&b={start_date.day}&c={start_date.year}&d={end_date.month}&e={end_date.day}&f={end_date.year}&g=d&api_token={eodhdkey}&fmt=json"
    
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        print(f"Error fetching historical data for {ticker}: {e}")
        return []

def get_all_historical_data(tickers: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get and combine historical data for multiple tickers"""
    all_data = []
    
    for ticker in tickers:
        hist_data = get_historical_data(ticker, start_date, end_date)
        if hist_data:
            df = pd.DataFrame(hist_data)
            df['ticker'] = ticker
            all_data.append(df)
    
    if not all_data:
        return pd.DataFrame()
        
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df['date'] = pd.to_datetime(combined_df['date'])
    
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
        price_data = hist_data.pivot(index='date', columns='ticker', values='close')
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
    
    analysis = try_llm_request(client, "gpt-4o-mini", messages)
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


def main():
    st.set_page_config(page_title="Automated Stock Analysis", layout="wide", page_icon="📈")
    
    # Initialize session state
    if 'search_results' not in st.session_state:
        st.session_state.search_results = None
    if 'all_stock_data' not in st.session_state:
        st.session_state.all_stock_data = None
    if 'stock_metadata' not in st.session_state:
        st.session_state.stock_metadata = None
    
    # Initial View
    st.title("Automated Stock Analysis")
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
                            "Current Price": f"${stock['realtime_data'].get('close', 0):.2f}",
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
    # if search_clicked and query:
    #     with st.spinner("Processing your request..."):
    #         debug_expander = st.expander("Debug Information", expanded=True)
            
    #         with debug_expander:
    #             st.write("**Search Process Debug Info**")
    #             enhanced_query = enhance_search_query_with_llm(query)
    #             st.code(f"Original Query: {query}")
    #             st.code(f"Enhanced Query: {enhanced_query}")
            
    #         # Get search results
    #         search_results = search_stocks_in_pinecone(enhanced_query)
    #         if not search_results:
    #             st.error("No matching stocks found. Please try a different search query.")
    #             return

    #         # Extract tickers and create a mapping of ticker to metadata
    #         stock_metadata = {}
    #         tickers = []
    #         for result in search_results:
    #             if 'Ticker' in result.metadata:
    #                 ticker = result.metadata['Ticker']
    #                 tickers.append(ticker)
    #                 stock_metadata[ticker] = result.metadata

    #         with debug_expander:
    #             st.write("Found Tickers:", tickers)

            try:
                # Get realtime data
                realtime_data = get_batch_realtime_data(tickers)
                print("\nRealtime Data:", realtime_data)  # Print to console instea
                
                # with debug_expander:
                #     st.write("Realtime Data:")
                #     st.write(realtime_data)
                
                # Collect all stock data
                all_stock_data = []
                for ticker in tickers:
                    metadata = stock_metadata[ticker]
                    yf_data = get_yfinance_data(ticker)
                    rt_data = next((item for item in realtime_data if item['code'].replace('.US', '') == ticker), {})
                    
                    stock_data = {
                        'ticker': ticker,
                        'name': metadata.get('Name', ''),
                        'description': metadata.get('Business Summary', ''),
                        'yfinance_data': yf_data,
                        'realtime_data': rt_data
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
                            'Current Price': stock['realtime_data'].get('close', 0),
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
                    'Current Price': stock['realtime_data'].get('close', 0),
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
if __name__ == "__main__":
    main()
