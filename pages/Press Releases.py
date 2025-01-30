import streamlit as st
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import numpy as np
import requests
from bs4 import BeautifulSoup
import pdfplumber
import re
 
class NewsDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len
 
    def __len__(self):
        return len(self.texts)
 
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
        }
 
def preprocess_text(text):
    """
    Preprocess text to better capture sustainability-related sentiment
    """
    # Clean the text
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
   
    # Extract key sections (focusing on important parts like headlines, conclusions)
    sections = text.split('\n')
    important_sections = []
   
    for section in sections:
        # Prioritize sections with key sustainability terms
        if any(term in section.lower() for term in [
            'sustainability', 'esg', 'environmental', 'social', 'governance',
            'green', 'renewable', 'sustainable', 'responsibility', 'achievement',
            'improvement', 'progress', 'success', 'excellence'
        ]):
            important_sections.append(section)
   
    # If we found important sections, use them; otherwise use full text
    processed_text = ' '.join(important_sections) if important_sections else text
   
    # Ensure we don't exceed BERT's maximum length
    return processed_text[:512]
 
@st.cache_resource
def load_model():
    try:
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=3
        )
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None
 
def predict_sentiment(model, tokenizer, text, device, max_len=256):
    model.eval()
   
    # Preprocess the text
    processed_text = preprocess_text(text)
   
    # Create dataset and dataloader
    dataset = NewsDataset([processed_text], tokenizer, max_len)
    loader = DataLoader(dataset, batch_size=1)
 
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
           
            # Apply additional weights for sustainability content
            adjusted_probs = adjust_probabilities_for_sustainability(processed_text, probs[0])
            sentiment = np.argmax(adjusted_probs)
            return sentiment, adjusted_probs
 
def adjust_probabilities_for_sustainability(text, probs):
    """
    Adjust sentiment probabilities based on sustainability-related content
    """
    # Define positive sustainability indicators
    positive_indicators = [
        'excellence', 'improvement', 'progress', 'achievement', 'success',
        'sustainable', 'renewable', 'responsible', 'positive', 'growth',
        'leadership', 'innovation', 'commitment', 'future'
    ]
   
    # Define negative indicators
    negative_indicators = [
        'risk', 'challenge', 'problem', 'issue', 'concern',
        'failure', 'decrease', 'decline', 'negative'
    ]
   
    text_lower = text.lower()
   
    # Count indicators
    positive_count = sum(text_lower.count(indicator) for indicator in positive_indicators)
    negative_count = sum(text_lower.count(indicator) for indicator in negative_indicators)
   
    # Adjust probabilities based on indicator counts
    adjusted_probs = probs.copy()
    if positive_count > negative_count:
        # Boost positive probability
        boost_factor = min(1.5, 1 + (positive_count - negative_count) * 0.1)
        adjusted_probs[2] *= boost_factor  # Increase positive sentiment
        # Normalize probabilities
        adjusted_probs = adjusted_probs / adjusted_probs.sum()
   
    return adjusted_probs
 
# [Rest of the code remains the same: adjust_esg_rating, extract_text_from_pdf, main, display_results]
 
def adjust_esg_rating(row, sentiment):
    rating = row['IVA_COMPANY_RATING']
    rating_levels = ["CCC", "B", "BB", "BBB", "A", "AA", "AAA"]
 
    if rating not in rating_levels:
        return rating
 
    current_index = rating_levels.index(rating)
   
    if sentiment == 0:
        adjustment = -1
    elif sentiment == 2:
        adjustment = 1
    else:
        adjustment = 0
 
    new_index = max(0, min(len(rating_levels) - 1, current_index + adjustment))
    new_rating = rating_levels[new_index]
 
    return new_rating
 
def scrape_article_content(article_url):
    try:
        response = requests.get(article_url)
        response.raise_for_status()
       
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        article_content = " ".join([para.get_text(strip=True) for para in paragraphs])
       
        return article_content
    except Exception as e:
        st.error(f"Error scraping article: {str(e)}")
        return None
 
def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            pages = [page.extract_text() for page in pdf.pages]
            return " ".join(pages).strip()
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None
 
def predict_esg_for_input(input_content, company_name, current_rating, dataset, model, tokenizer, device):
    company_data = dataset[dataset['COMPANY_NAME'] == company_name]
 
    if company_data.empty:
        raise ValueError(f"No data found for the company: {company_name}")
 
    text_analysis = company_data.iloc[0]['IVA_RATING_ANALYSIS'] + " " + company_data.iloc[0]['ESG_HEADLINE']
   
    # Add context about sustainability report
    context = "This is a sustainability report highlighting company achievements and progress. "
    sentiment, sentiment_probs = predict_sentiment(model, tokenizer, context + text_analysis + " " + input_content, device)
 
    updated_rating = adjust_esg_rating(company_data.iloc[0], sentiment)
 
    return {
        "company_name": company_name,
        "current_rating": current_rating,
        "new_rating": updated_rating,
        "sentiment": sentiment,
        "sentiment_probs": sentiment_probs,
        "extracted_text": input_content
    }
 
def main():
    st.title("ESG Sentiment Analysis and Rating Predictor")
   
    # Load model and set device
    model, tokenizer = load_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
 
    # Initialize session state for storing analysis results
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
 
    # File uploader for Excel dataset
    st.subheader("1. Upload ESG Dataset")
    uploaded_file = st.file_uploader("Upload Excel file with ESG data", type=['xlsx'])
   
    if uploaded_file is not None:
        esg_data = pd.read_excel(uploaded_file)
        esg_data['IVA_RATING_ANALYSIS'] = esg_data['IVA_RATING_ANALYSIS'].fillna("")
        esg_data['ESG_HEADLINE'] = esg_data['ESG_HEADLINE'].fillna("")
       
        # Company selection
        st.subheader("2. Select Company and Current Rating")
        companies = esg_data['COMPANY_NAME'].unique()
        company_name = st.selectbox("Select Company", companies)
        current_rating = st.selectbox("Current ESG Rating", ["CCC", "B", "BB", "BBB", "A", "AA", "AAA"])
       
        # Input method selection
        st.subheader("3. Choose Input Method")
        input_method = st.radio("Select Input Method", ["PDF Upload", "Article URL"])
       
        content = None
       
        if input_method == "PDF Upload":
            uploaded_pdf = st.file_uploader("Upload PDF file", type=['pdf'])
            if uploaded_pdf is not None:
                content = extract_text_from_pdf(uploaded_pdf)
                if content:
                    st.success("PDF processed successfully!")
       
        else:  # Article URL
            article_url = st.text_input("Enter Article URL")
            if article_url:
                content = scrape_article_content(article_url)
                if content:
                    st.success("Article scraped successfully!")
 
        # Add a clear "Run Analysis" button
        st.subheader("4. Run Analysis")
        if st.button("Run Analysis", key="run_analysis"):
            if content:
                with st.spinner("Analyzing content..."):
                    try:
                        result = predict_esg_for_input(
                            content, company_name, current_rating,
                            esg_data, model, tokenizer, device
                        )
                        st.session_state.analysis_results = result
                        st.success("Analysis completed!")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
            else:
                st.warning("Please ensure content is loaded (either PDF or Article URL) before running analysis.")
 
        # Display results if available
        if st.session_state.analysis_results:
            display_results(st.session_state.analysis_results)
 
def display_results(result):
    st.header("Analysis Results")
   
    # Create three columns for the ratings
    col1, col2, col3 = st.columns(3)
   
    with col1:
        st.metric(
            "Current Rating",
            result["current_rating"],
            delta=None,
            help="Original ESG rating"
        )
    with col2:
        delta = None
        if result["new_rating"] != result["current_rating"]:
            delta = f"Changed from {result['current_rating']}"
        st.metric(
            "New Rating",
            result["new_rating"],
            delta=delta,
            help="Adjusted ESG rating based on sentiment analysis"
        )
    with col3:
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        st.metric(
            "Overall Sentiment",
            sentiment_map[result["sentiment"]],
            help="Predicted sentiment from content analysis"
        )
   
    # Display sentiment probabilities
    st.subheader("Sentiment Probability Distribution")
    prob_df = pd.DataFrame({
        'Sentiment': ['Negative', 'Neutral', 'Positive'],
        'Probability': result["sentiment_probs"] * 100  # Convert to percentage
    })
    prob_df['Probability'] = prob_df['Probability'].round(2)
    # Create a more visually appealing bar chart
    chart = st.bar_chart(
        prob_df.set_index('Sentiment'),
        use_container_width=True,
        height=400
    )
   
    # Add a table with exact values
    st.table(prob_df.set_index('Sentiment').style.format({'Probability': '{:.2f}%'}))
   
    # Display extracted text
    with st.expander("View Analyzed Text"):
        st.text_area("", result["extracted_text"], height=300)
 
    # Add a clear button to reset the analysis
    if st.button("Clear Results"):
        st.session_state.analysis_results = None
        st.experimental_rerun()
 
if __name__ == "__main__":
    main()