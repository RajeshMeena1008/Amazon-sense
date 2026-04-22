"""
Train sentiment analysis model on Amazon reviews data.
Saves model artifacts to the models/ directory.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, 
    confusion_matrix, roc_auc_score
)
import re


def clean_text(text):
    """Basic text cleaning."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_features(df):
    """Extract engineered features from the dataframe."""
    features = pd.DataFrame()
    features['rating'] = df['rating']
    features['helpful_votes'] = df['helpful_votes'].fillna(0)
    features['verified_purchase'] = df['verified_purchase'].astype(int)
    features['review_length'] = df['review_text'].apply(lambda x: len(str(x).split()))
    
    # Category encoding
    le = LabelEncoder()
    features['category_encoded'] = le.fit_transform(df['category'].fillna('Unknown'))
    
    return features, le


def train_model():
    print("Loading dataset...")
    df = pd.read_csv("data/amazon_reviews.csv")
    df['cleaned_review'] = df['review_text'].apply(clean_text)
    
    X_text = df['cleaned_review']
    X_meta, label_encoder = extract_features(df)
    y = df['liked']
    
    # Split
    (X_text_train, X_text_test, 
     X_meta_train, X_meta_test, 
     y_train, y_test) = train_test_split(
        X_text, X_meta, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words='english'
    )
    X_text_train_vec = tfidf.fit_transform(X_text_train)
    X_text_test_vec = tfidf.transform(X_text_test)
    
    # Combine TF-IDF + meta features
    from scipy.sparse import hstack, csr_matrix
    X_train_combined = hstack([X_text_train_vec, csr_matrix(X_meta_train.values)])
    X_test_combined = hstack([X_text_test_vec, csr_matrix(X_meta_test.values)])
    
    print("Training models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    best_model = None
    best_accuracy = 0
    results = {}
    
    for name, model in models.items():
        model.fit(X_train_combined, y_train)
        preds = model.predict(X_test_combined)
        acc = accuracy_score(y_test, preds)
        results[name] = acc
        print(f"  {name}: {acc:.4f}")
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = (name, model)
    
    print(f"\nBest model: {best_model[0]} ({best_accuracy:.4f})")
    
    final_model = best_model[1]
    final_preds = final_model.predict(X_test_combined)
    
    print("\nClassification Report:")
    print(classification_report(y_test, final_preds, target_names=["Disliked", "Liked"]))
    
    # Save artifacts
    os.makedirs("models", exist_ok=True)
    
    artifacts = {
        "model": final_model,
        "tfidf": tfidf,
        "label_encoder": label_encoder,
        "model_name": best_model[0],
        "accuracy": best_accuracy,
        "all_results": results,
        "categories": list(df['category'].unique()),
        "products": list(df['product_name'].unique()),
    }
    
    with open("models/sentiment_model.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    
    print("\nModel artifacts saved to models/sentiment_model.pkl")
    return artifacts


if __name__ == "__main__":
    train_model()
