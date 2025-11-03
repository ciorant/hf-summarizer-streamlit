# AI News Summarizer - End-to-End ML Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://english-news-summarizer.streamlit.app)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue)](https://huggingface.co/ciorant/news-summarizer)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> An end-to-end machine learning project that fine-tunes BART-Large-CNN for news article summarization, complete with a user-friendly Streamlit interface and comprehensive training pipeline.

## Project Overview

This project demonstrates a complete ML workflow from data preprocessing to model deployment:

- **Fine-tuned BART-Large-CNN model** for news summarization
- **Custom training pipeline** with robust error handling and overflow prevention
- **Interactive Streamlit web app** with customizable generation parameters
- **Production-ready deployment** on Hugging Face Hub with API access

## Live Demo

Try the model yourself:
- **Web App**: [Streamlit Demo](https://english-news-summarizer.streamlit.app)
- **API**: [Hugging Face Model](https://huggingface.co/ciorant/news-summarizer)

## Model Performance

| Metric | Score |
|--------|-------|
| ROUGE-1 | 0.42 |
| ROUGE-2 | 0.21 |
| ROUGE-L | 0.29 |
| Training Steps | 4,000 |
| Dataset | CNN/DailyMail 3.0.0 |
| Base Model | facebook/bart-large-cnn |

## Technical Stack

### Model & Data
- **Base Model**: `facebook/bart-large-cnn`
- **Dataset**: CNN/DailyMail 3.0.0 (287k+ article-summary pairs)
- **Framework**: Transformers, PyTorch
- **Training**: Seq2SeqTrainer with custom metrics

### Key Features
- **Robust error handling** for tokenization overflow issues
- **Custom ROUGE metrics** with fallback mechanisms  
- **Checkpoint resuming** from Google Drive
- **Memory optimization** for Colab (FP16, gradient accumulation)
- **Advanced generation controls** (beam search, sampling, penalties)

### Project structure

hf-news-summarizer/   
├── news-summarizer.ipynb   - complete training pipeline   
├── streamlit_app/          - Streamlit app folder   
│   ├── app.py              - Streamlit interface   
│   └── requirements.txt    - app dependencies   
├── .devcontainer/          - configuration for VS Code Dev Containers / GitHub Codespaces   
└── README.md               - this file   

### Data Preprocessing

- Loaded CNN/DailyMail 3.0.0 dataset (287k article-summary pairs) and picked a subset of those-
- Tokenized with BART tokenizer (max 512 tokens input, 256 output)
- Train/validation split for evaluation

### Training Configuration

- **Batch Size**: 2 per device with 4 gradient accumulation steps
- **Learning Rate**: 3e-5 with 500 warmup steps
- **Epochs**: around 1.5 epochs (early-stopped for picking the best scores)
- **Optimization**: AdamW with 0.01 weight decay
- **Mixed Precision**: FP16 for memory efficiency

## Streamlit App Features
### User Controls

- **Length Settings**: Adjustable min/max summary length
- **Quality Controls**: Beam search, length penalty, repetition penalty
- **Advanced Options**: Temperature, sampling, generation parameters
- **Real-time Metrics**: Word count, compression ratio, generation time

## Acknowledgments

- **Dataset**: CNN/DailyMail dataset creators
- **Base Model**: Facebook's BART-Large-CNN team
- **Framework**: Hugging Face Transformers library
- **Inspiration**: Modern NLP summarization research





