# StockPicker

StockPicker is an intelligent stock analysis tool that helps users discover and analyze stocks based on natural language queries. The application uses advanced language models to understand user queries and provides comprehensive stock analysis including financial metrics, market trends, and comparative visualizations.

## Features

- **Natural Language Search**: Enter plain English descriptions to find relevant stocks
- **Comprehensive Stock Analysis**: View detailed financial metrics, market performance, and business summaries
- **Interactive Visualizations**:
  - Price comparison charts with customizable date ranges
  - Market trend radar showing relative performance across key metrics
  - Factor-based analysis with detailed explanations
- **Data Export**: Download stock information and historical data as CSV files

## Tech Stack

- Streamlit for the web interface
- Groq, Gemini, OpenAI for query enhancement and analysis
- Pinecone for vector similarity search
- YFinance for stock data
- Plotly for interactive visualizations

## Installation

1. Clone the repository:

2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate  
   #On Windows: venv\Scripts\activate

3. Install required packages:
   pip install -r requirements.txt

4. Set up environment variables in a .env file:
   OPENAI_API_KEY=your_openai_api_key
   GROQ_API_KEY=your_groq_api_key
   GEMINI_API_KEY=your_gemini_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   PINECONE_INDEX=your_pinecone_index_name
   PINECONE_ENV=your_pinecone_env

## Usage

1.Run the Streamlit app:
streamlit run main.py

2.Enter a description of the stocks you're interested in (e.g., "electric car manufacturers" or "hotels")

3.Explore the analysis, visualizations, and download data as needed
