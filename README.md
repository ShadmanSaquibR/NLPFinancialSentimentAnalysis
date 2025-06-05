# Financial News & Earnings Call Sentiment Analyzer

A real-time sentiment analysis tool for financial documents using the FinBERT NLP model, providing visual insights into market sentiment from earnings calls and news articles.

## Features

- 📝 **Text Analysis**: Paste any financial text (earnings call transcripts, news articles, reports)
- ⚡ **Real-time Processing**: Analyzes content sentence-by-sentence using FinBERT
- 📊 **Interactive Visuals**: Donut chart visualization of sentiment distribution
- 📈 **Quantitative Metrics**: Breakdown of sentiment percentages and sentence counts
- ⏱️ **Performance Tracking**: Displays analysis execution time

## How It Works

1. Paste financial text into the input box
2. Click "Run" to initiate analysis
3. View sentiment distribution through a color-coded donut chart
4. Examine detailed metrics for positive, neutral, and negative sentiments
5. Access model information and source code via the sidebar

## Installation

1. Clone repository:
```bash
git clone https://github.com/ShadmanSaquibR/SentimentAnalyserFinancialNewsandEarningsCall.git
cd SentimentAnalyserFinancialNewsandEarningsCall

# Install dependencies
pip install -r requirements.txt

# Download NLTK resources
python -c "import nltk; nltk.download('punkt')"

# Usage
streamlit run streamlit_app.py