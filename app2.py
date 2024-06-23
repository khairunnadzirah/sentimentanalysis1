import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np
import pandas as pd
from datetime import datetime
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import io
from fpdf import FPDF

# Define the new group mappings
label_mapping = {
    1: 0,  # 'worse' -> 'negative'
    2: 0,  # 'bad' -> 'negative'
    3: 1,  # 'neutral'
    4: 2,  # 'good' -> 'positive'
    5: 2   # 'excellent' -> 'positive'
}

# New id2label and label2id for 3 categories
id2label = {0: "negative", 1: "neutral", 2: "positive"}

# Load model and tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Prediction function
def predict(text):
    # Tokenize the input
    encoded_input = tokenizer(text, return_tensors='pt')
    
    # Move input tensors to the same device as the model
    encoded_input = {key: value.to(device) for key, value in encoded_input.items()}
    
    # Get model output
    with torch.no_grad():
        output = model(**encoded_input)
    
    # Accessing the logits
    logits = output.logits.detach().cpu().numpy()
    scores = softmax(logits, axis=1)[0]
    
    # Get the predicted label
    predicted_label = np.argmax(scores)
    
    return id2label[predicted_label], scores

# Streamlit app
st.title("E-commerce Sentiment Analysis App")

# Custom CSS for button styling, centering content, and placeholder text opacity
st.markdown(
    """
    <style>
    div.stButton > button {
        background-color: #4CAF50;
        color: white;
        border: 1px solid white;
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    div.stButton > button:hover {
        background-color: #45a049;
        color: white;
        border: 1px solid white;
    }
    .centered-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-top: 20px;
    }
    .large-emoji {
        font-size: 50px;
        margin-top: 10px;
    }
    .bold-text {
        font-size: 24px;
        font-weight: bold;
    }
    .centered-textarea {
        display: flex;
        justify-content: center;
        margin-bottom: 10px;
    }
    .centered-textarea > div {
        width: 100%;
        max-width: 600px;
    }
    ::placeholder {
        color: rgba(0, 0, 0, 0.5); /* Decreased opacity for placeholder text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Define emoticons
emoticons = {
    "default": "❓",
    "negative": "😞",
    "neutral": "😐",
    "positive": "😊"
}

# Filters
st.header("Filters")
analysis_type = st.radio("Select Analysis Type:", ["Enter a text for sentiment analysis:", "Enter a file for sentiment analysis:"])

# Centered input area and analyze button
with st.container():
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    if analysis_type == "Enter a text for sentiment analysis:":
        # User input
        st.markdown('<div class="centered-textarea">', unsafe_allow_html=True)
        user_input = st.text_area("", placeholder="Enter a text for sentiment analysis")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # File input
        uploaded_file = st.file_uploader("Enter a file for sentiment analysis (CSV format with columns 'Review' and 'Date'):")

    # Button to trigger analysis
    analyze_button = st.button("Analyze")
    st.markdown('</div>', unsafe_allow_html=True)

# Pre-render the Predictive Sentiment section below the "Analyze" button
sentiment_placeholder = st.empty()
date_placeholder = st.empty()
confidence_placeholder = st.empty()
table_placeholder = st.empty()
summary_chart_placeholder = st.empty()
trend_chart_placeholder = st.empty()
wordcloud_placeholder = st.empty()
report_button_placeholder = st.empty()

# Default display before analysis
if analysis_type == "Enter a text for sentiment analysis:":
    sentiment_placeholder.markdown(
        f'<div class="centered-content"><div class="bold-text">Predictive Sentiment:</div>'
        f'<div class="large-emoji">{emoticons["default"]}</div></div>', 
        unsafe_allow_html=True)
    date_placeholder.markdown(
        f'<div class="centered-content">Date: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</div>', 
        unsafe_allow_html=True)
    confidence_placeholder.markdown(
        '<div class="centered-content">Confidence Score: 0.0000</div>', 
        unsafe_allow_html=True)

# Function to generate a word cloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    return fig

# Function to save a plotly figure as an image
def save_plotly_fig_as_image(fig, filename):
    fig.write_image(filename)

# Function to save a matplotlib figure as an image
def save_matplotlib_fig_as_image(fig, filename):
    fig.savefig(filename, bbox_inches='tight')

# Function to generate a PDF report
def generate_pdf_report(result_df, sentiment_summary, trend_fig, wordcloud_fig):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Title
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(200, 10, "Sentiment Analysis Report", ln=True, align='C')

    # Add table of sentiment analysis results
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, "Sentiment Analysis Results:", ln=True)
    pdf.ln(5)

    # Add table header
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(40, 10, "Review", 1)
    pdf.cell(30, 10, "Date", 1)
    pdf.cell(60, 10, "Prediction Sentiment", 1)
    pdf.cell(60, 10, "Confidence Score", 1)
    pdf.ln()

    # Add table rows
    pdf.set_font("Arial", size=12)
    for index, row in result_df.iterrows():
        pdf.cell(40, 10, str(row['Review']), 1)
        pdf.cell(30, 10, str(row['Date']), 1)
        pdf.cell(60, 10, str(row['Prediction Sentiment']), 1)
        pdf.cell(60, 10, str(row['Confidence Score']), 1)
        pdf.ln()

    # Add sentiment summary
    pdf.ln(10)
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Sentiment Summary:", ln=True)
    pdf.ln(5)
    for index, row in sentiment_summary.iterrows():
        pdf.cell(100, 10, f"{row['Sentiment']}: {row['Count']}", ln=True)

    # Add sentiment trends
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Sentiment Analysis Trends Over Time:", ln=True)
    pdf.ln(5)

    # Save sentiment trend plot as image and add to PDF
    trend_fig_file = "trend_fig.png"
    save_plotly_fig_as_image(trend_fig, trend_fig_file)
    pdf.image(trend_fig_file, x=10, y=None, w=190)

    # Add word cloud
    pdf.add_page()
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(200, 10, "Word Cloud:", ln=True)
    pdf.ln(5)

    # Save word cloud as image and add to PDF
    wordcloud_fig_file = "wordcloud_fig.png"
    save_matplotlib_fig_as_image(wordcloud_fig, wordcloud_fig_file)
    pdf.image(wordcloud_fig_file, x=10, y=None, w=190)

    # Save PDF to a bytes buffer
    pdf_buffer = io.BytesIO()
    pdf.output(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

# Initialize session state for result storage
if 'result_df' not in st.session_state:
    st.session_state['result_df'] = None
if 'sentiment_summary' not in st.session_state:
    st.session_state['sentiment_summary'] = None
if 'trend_fig' not in st.session_state:
    st.session_state['trend_fig'] = None
if 'wordcloud_fig' not in st.session_state:
    st.session_state['wordcloud_fig'] = None

# Prediction and display results
if analyze_button:
    if analysis_type == "Enter a text for sentiment analysis:" and user_input:
        text_to_analyze = user_input
        label, scores = predict(text_to_analyze)
        
        # Update result with larger emoticon
        sentiment_placeholder.markdown(
            f'<div class="centered-content"><div class="bold-text">Predictive Sentiment:</div>'
            f'<div class="large-emoji">{emoticons[label]}</div></div>', 
            unsafe_allow_html=True)
        date_placeholder.markdown(
            f'<div class="centered-content">Date: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</div>', 
            unsafe_allow_html=True)
        confidence_placeholder.markdown(
            f'<div class="centered-content">Confidence Score: {np.max(scores):.4f}</div>', 
            unsafe_allow_html=True)
    elif analysis_type == "Enter a file for sentiment analysis:" and uploaded_file:
        try:
            df = pd.read_csv(uploaded_file, delimiter=",", on_bad_lines='skip')
            if 'Review' in df.columns and 'Date' in df.columns:
                results = []
                for index, row in df.iterrows():
                    review = row['Review']
                    date = row['Date']
                    label, scores = predict(review)
                    results.append({
                        "Review": review,
                        "Date": date,
                        "Prediction Sentiment": label,
                        "Confidence Score": np.max(scores)
                    })
                result_df = pd.DataFrame(results)
                
                # Store the result in session state
                st.session_state['result_df'] = result_df
                
                # Display the DataFrame with a scrollbar
                table_placeholder.dataframe(result_df, height=400)
                
                # Convert date column to datetime
                result_df['Date'] = pd.to_datetime(result_df['Date'])

                # Sentiment summary as a bar chart
                sentiment_summary = result_df['Prediction Sentiment'].value_counts().reset_index()
                sentiment_summary.columns = ['Sentiment', 'Count']
                fig_summary = px.bar(sentiment_summary, x='Sentiment', y='Count', title='Sentiment Summary')
                summary_chart_placeholder.plotly_chart(fig_summary)
                
                # Store the sentiment summary in session state
                st.session_state['sentiment_summary'] = sentiment_summary

                # Create a sentiment trend plot
                trend_df = result_df.groupby(['Date', 'Prediction Sentiment']).size().reset_index(name='Counts')
                fig_trend = px.line(trend_df, x='Date', y='Counts', color='Prediction Sentiment',
                              title='Sentiment Analysis Trends Over Time')
                trend_chart_placeholder.plotly_chart(fig_trend)
                
                # Store the trend figure in session state
                st.session_state['trend_fig'] = fig_trend

                # Generate word cloud
                all_reviews = ' '.join(result_df['Review'])
                wordcloud_fig = generate_wordcloud(all_reviews)
                wordcloud_placeholder.pyplot(wordcloud_fig)
                
                # Store the word cloud figure in session state
                st.session_state['wordcloud_fig'] = wordcloud_fig

                # Titles and Download button
                st.markdown("<h3 style='text-align: center;'>Word Cloud</h3>", unsafe_allow_html=True)
                wordcloud_placeholder.pyplot(wordcloud_fig)
                st.markdown("<h3 style='text-align: center;'>Generate Report</h3>", unsafe_allow_html=True)
                report_button_placeholder.markdown('<div class="centered-content">', unsafe_allow_html=True)
                if report_button_placeholder.button("Download"):
                    if st.session_state['result_df'] is not None:
                        pdf_report = generate_pdf_report(st.session_state['result_df'], st.session_state['sentiment_summary'], st.session_state['trend_fig'], st.session_state['wordcloud_fig'])
                        st.download_button(label="Download Report", data=pdf_report, file_name="sentiment_analysis_report.pdf")
                report_button_placeholder.markdown('</div>', unsafe_allow_html=True)

            else:
                st.write("The uploaded file does not have the required columns 'Review' and 'Date'.")
        except pd.errors.ParserError as e:
            st.error(f"Error parsing the file: {e}")
    else:
        st.write("Please enter some text or upload a file for analysis.")
