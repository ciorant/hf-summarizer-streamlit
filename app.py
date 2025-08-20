import streamlit as st
from transformers import pipeline
import re

# --- Helper: Clean summary text ---
def clean_summary(text: str) -> str:
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- Load summarization pipeline ---
@st.cache_resource
def load_summarizer():
    # ✅ Replace "facebook/bart-large-cnn" with your own model on HF Hub
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# --- Streamlit App ---
st.title("Text Summarizer with Cleanup ✨")

user_input = st.text_area("Paste text to summarize:")

if st.button("Summarize"):
    if user_input
