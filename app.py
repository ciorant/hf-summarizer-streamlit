
import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re
import time

# --- Configuration ---
st.set_page_config(
    page_title="News Summarizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Helper Functions ---
def clean_summary(text: str) -> str:
    """Clean summary text by fixing spacing and formatting"""
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def estimate_reading_time(text: str) -> int:
    """Estimate reading time in minutes (assuming 200 words per minute)"""
    word_count = len(text.split())
    return max(1, round(word_count / 200))

# --- Load Model with Optimizations ---
@st.cache_resource
def load_model_and_tokenizer():
    """Load model with optimizations for speed"""
    model_name = "facebook/bart-large-cnn"  # Replace with your model
    
    # Load with optimizations
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    return model, tokenizer

def create_summarizer(model, tokenizer):
    """Create optimized pipeline"""
    return pipeline(
        "summarization",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

# --- Initialize Model ---
with st.spinner("Loading AI model... (This may take a moment on first run)"):
    model, tokenizer = load_model_and_tokenizer()
    summarizer = create_summarizer(model, tokenizer)

# --- Sidebar Controls ---
st.sidebar.header("📊 Summary Controls")

# Length controls
st.sidebar.subheader("Length Settings")
min_length = st.sidebar.slider(
    "Minimum length (words)",
    min_value=10,
    max_value=100,
    value=30,
    step=5,
    help="Minimum number of words in summary"
)

max_length = st.sidebar.slider(
    "Maximum length (words)",
    min_value=50,
    max_value=300,
    value=130,
    step=10,
    help="Maximum number of words in summary"
)

# Generation parameters
st.sidebar.subheader("Quality Settings")
num_beams = st.sidebar.selectbox(
    "Beam search beams",
    options=[2, 3, 4, 5],
    index=2,
    help="More beams = better quality but slower"
)

length_penalty = st.sidebar.slider(
    "Length penalty",
    min_value=0.5,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Higher = encourages longer summaries"
)

# Advanced options
st.sidebar.subheader("Advanced Options")
do_sample = st.sidebar.checkbox(
    "Enable sampling",
    value=False,
    help="Enable for more creative but less predictable summaries"
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Higher = more creative (only if sampling enabled)",
    disabled=not do_sample
)

repetition_penalty = st.sidebar.slider(
    "Repetition penalty",
    min_value=1.0,
    max_value=2.0,
    value=1.2,
    step=0.1,
    help="Higher = less repetitive text"
)

# --- Main App ---
st.title("📰 AI News Summarizer")
st.markdown("Transform long articles into concise, readable summaries with customizable parameters.")

# Input section
st.subheader("📝 Input Article")
col1, col2 = st.columns([3, 1])

with col1:
    user_input = st.text_area(
        "Paste your news article here:",
        height=200,
        placeholder="Paste a news article in English...",
        help="For best results, use articles between 100-2000 words"
    )

with col2:
    if user_input.strip():
        word_count = len(user_input.split())
        reading_time = estimate_reading_time(user_input)
        
        st.metric("Word Count", word_count)
        st.metric("Est. Reading Time", f"{reading_time} min")
        
        # Input validation
        if word_count < 50:
            st.warning("⚠️ Article seems short. Consider adding more content.")
        elif word_count > 2000:
            st.warning("⚠️ Very long article. Processing may be slow.")

# Example articles
with st.expander("📚 Try Example Articles"):
    col1, col2, col3 = st.columns(3)
    
    example1 = """Technology giant Apple announced today a groundbreaking new iPhone model that features revolutionary AI capabilities and extended battery life. The iPhone 16 Pro includes advanced machine learning processors and can operate for up to 72 hours on a single charge. Industry experts believe this could reshape the smartphone market entirely."""
    
    example2 = """The Federal Reserve announced a surprise interest rate cut of 0.5% yesterday, citing concerns about global economic uncertainty. Stock markets responded positively, with the Dow Jones rising 400 points. Economists predict this move could stimulate consumer spending and business investment in the coming quarters."""
    
    example3 = """Scientists at MIT have developed a new solar panel technology that is 40% more efficient than traditional panels. The breakthrough uses quantum dots to capture a wider spectrum of light. This innovation could significantly reduce the cost of renewable energy and accelerate the transition to clean power."""
    
    with col1:
        if st.button("Tech News", use_container_width=True):
            st.rerun()
    with col2:
        if st.button("Economics", use_container_width=True):
            st.rerun()  
    with col3:
        if st.button("Science", use_container_width=True):
            st.rerun()

# Summarization
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    summarize_button = st.button("🚀 Generate Summary", type="primary", use_container_width=True)

if summarize_button and user_input.strip():
    # Validation
    if max_length <= min_length:
        st.error("❌ Maximum length must be greater than minimum length!")
    else:
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Preprocessing
            status_text.text("🔄 Processing article...")
            progress_bar.progress(25)
            
            # Truncate input if too long (models have token limits)
            input_tokens = tokenizer.encode(user_input, truncation=True, max_length=1024)
            if len(input_tokens) == 1024:
                st.info("ℹ️ Article was truncated to fit model limits.")
            
            # Generate summary
            status_text.text("🤖 AI is generating summary...")
            progress_bar.progress(50)
            
            start_time = time.time()
            
            summary_params = {
                "max_length": max_length,
                "min_length": min_length,
                "num_beams": num_beams,
                "length_penalty": length_penalty,
                "repetition_penalty": repetition_penalty,
                "early_stopping": True,
                "do_sample": do_sample
            }
            
            if do_sample:
                summary_params["temperature"] = temperature
            
            raw_summary = summarizer(user_input, **summary_params)
            
            progress_bar.progress(75)
            status_text.text("🎨 Polishing summary...")
            
            # Post-process
            summary_text = raw_summary[0]['summary_text']
            final_summary = clean_summary(summary_text)
            
            generation_time = time.time() - start_time
            
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            # Results
            st.subheader("📋 Summary")
            
            # Summary stats
            summary_words = len(final_summary.split())
            compression_ratio = round((1 - summary_words/len(user_input.split())) * 100, 1)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Summary Words", summary_words)
            col2.metric("Compression", f"{compression_ratio}%")
            col3.metric("Generation Time", f"{generation_time:.1f}s")
            col4.metric("Reading Time", f"{estimate_reading_time(final_summary)} min")
            
            # Display summary
            st.markdown(f"""
            <div style="
                background-color: #f0f9ff;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #3b82f6;
                margin: 20px 0;
            ">
                <h4 style="color: #1e40af; margin-bottom: 15px;">📄 Generated Summary</h4>
                <p style="font-size: 16px; line-height: 1.6; margin: 0; color: #1f2937;">{final_summary}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Download option
            st.download_button(
                label="📥 Download Summary",
                data=f"Original Article ({len(user_input.split())} words):\n{user_input}\n\n" +
                     f"Generated Summary ({summary_words} words):\n{final_summary}",
                file_name="news_summary.txt",
                mime="text/plain"
            )
            
        except Exception as e:
            st.error(f"❌ Error generating summary: {str(e)}")
            st.info("💡 Try reducing the article length or adjusting the parameters.")
        
        finally:
            progress_bar.empty()
            status_text.empty()

elif summarize_button:
    st.warning("⚠️ Please enter an article to summarize!")

# --- Footer ---
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p>Built with ❤️ using Streamlit and Transformers</p>
        <p><small>Tip: Use the sidebar controls to customize your summaries!</small></p>
    </div>
    """,
    unsafe_allow_html=True)
