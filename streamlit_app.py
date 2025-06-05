import streamlit as st #https://www.youtube.com/watch?v=vzlQkAzWCeI, type streamlit run Streamlitcopy.py in terminal to run the app

# Import necessary libraries
import re
import nltk
from transformers import pipeline
import pandas as pd
import plotly.express as px
import time  # Import time module for timing
 
st.title(":red[Financial News and Earnings Call Sentiment Analyser]")
full_text=st.text_area("_Paste text you want analysed (earnings call transcripts, news article, etc.)_", height=80)

if st.button("Run",type='primary'):
    start_time = time.time()
    full_text = re.sub(r'\s+',' ', full_text)  # Remove excessive whitespace
    full_text = re.sub(r'[^\w\s.,;:!?]','', full_text)  # Remove special characters

    #Using high-level FinBERT to analyse extracted text
    from nltk.tokenize import sent_tokenize

    # Split text into sentences
    sentences = sent_tokenize(full_text)

    # Process each sentence individually
    # Use a pipeline as a high-level helper
    pipe = pipeline("text-classification", model="ProsusAI/finbert")
    sentence_results = pipe(sentences)

    # Aggregate sentiment scores
    sentiment_counts = {'positive': 0,'neutral': 0,'negative': 0}
    sentiment_scores = {'positive': 0,'neutral': 0,'negative': 0}  
    for result in sentence_results:
        label = result['label']
        score = result['score']
        sentiment_counts[label] += 1
        sentiment_scores[label] += score
    end_time = time.time()
    elapsed_time = end_time - start_time
    st.success(f'Analysis completed in {elapsed_time:.2f} seconds')

    # Create pie chart
    placeholder = st.empty()
    total = len(sentences)
    if total > 0:
        st.subheader("Sentiment Distribution")

        # Prepare data for visualization
        df = pd.DataFrame({
            'Sentiment': ['Positive', 'Neutral', 'Negative'],
            'Count': [
                sentiment_counts['positive'],
                sentiment_counts['neutral'],
                sentiment_counts['negative']
            ]
        })
        
        # Creating colorful pie chart with Plotly
        fig = px.pie(
            df,
            names='Sentiment',
            values='Count',
            color='Sentiment',
            color_discrete_map={
                'Positive': '#00fa9a',   # Green
                'Neutral': '#F5F5F5',    # Light Gray
                'Negative': '#de403f'    # Red
            },
            hole=0.6, # Creates a donut chart
            labels={'Count': 'Number of Sentences'},
        )
        
        # Enhance visual styling
        fig.update_traces(
            textposition='inside',
            textinfo='label'
        )
        fig.update_layout(
            showlegend=False,
            font=dict(size=14),
            margin=dict(t=30, b=0, l=0, r=0)
        )
        
        # Organize layout with columns 
        col1, col2 = st.columns([3,1], gap='medium')
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.metric("**Total Sentences**", total)
            st.write("**Detailed Breakdown:**")
            for label, count in sentiment_counts.items():
                st.write(f"{label.capitalize()}: {count} ({count/total:.1%})")
    else:
        placeholder.empty()
        
with st.sidebar:
    st.title("About")
    st.markdown("This app uses <a href='https://huggingface.co/ProsusAI/finbert' style='color: #f14646;' target='_blank'>FinBERT</a> NLP model to analyze the sentiment of financial news articles and earnings call transcripts within seconds", unsafe_allow_html=True)

    st.caption("Developed by Shadman Saquib Rahman")  
    st.caption("_The source code available on [GitHub](https://github.com/ShadmanSaquibR/SentimentAnalyserFinancialNewsandEarningsCall)_")
nltk.download('punkt_tab')
