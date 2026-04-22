#!/bin/bash
# Auto-setup script for Streamlit Cloud deployment
# Generates dataset and trains the model if not already done

echo "🔄 Setting up Amazon Sentiment Analyzer..."

if [ ! -f "data/amazon_reviews.csv" ]; then
    echo "📦 Generating dataset..."
    python data/generate_data.py
fi

if [ ! -f "models/sentiment_model.pkl" ]; then
    echo "🤖 Training ML model..."
    python train_model.py
fi

echo "✅ Setup complete! Starting app..."
