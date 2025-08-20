import streamlit as st
from transformers import pipeline
import re

# --- Helper: Clean summary text ---
def clean_summary(text: str) -> str:
    # Remove spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Load summarization pipeline ---
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")  # or your HF model

summarizer = load_summarizer()

# --- Streamlit App ---
st.title("News Summarizer 😎")

user_input = st.text_area("Paste news article in English to summarize:")

if st.button("Summarize"):
    if user_input.strip():
        # Run model
        raw_summary = summarizer(user_input, max_length=150, min_length=30, do_sample=False)
        summary_text = raw_summary[0]['summary_text']
        
        # Clean summary
        final_summary = clean_summary(summary_text)
        
        st.subheader("Summary")
        st.write(final_summary)
