# 🛒 Amazon Product Sentiment Analyzer

An AI-powered Streamlit web app that predicts whether a customer **liked or disliked** an Amazon product based on their review text, star rating, and other metadata.

---

## 📊 Model Performance

| Model | Accuracy |
|---|---|
| **Logistic Regression** | **95.25%** ✅ Best |
| Gradient Boosting | 95.00% |
| Random Forest | 94.75% |

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset
```bash
python data/generate_data.py
```

### 3. Train the Model
```bash
python train_model.py
```

### 4. Run the App
```bash
streamlit run app.py
```

---

## 🗂️ Project Structure

```
amazon_sentiment/
├── app.py                  ← Main Streamlit app
├── train_model.py          ← Model training script
├── requirements.txt        ← Python dependencies
├── data/
│   ├── generate_data.py    ← Dataset generator
│   └── amazon_reviews.csv  ← Generated dataset (2,000 reviews)
└── models/
    └── sentiment_model.pkl ← Trained model artifacts
```

---

## 🧠 How It Works

1. **Data** – 2,000 Amazon-style reviews with ratings, categories, and metadata
2. **Text Features** – TF-IDF vectorizer (5,000 features, unigrams + bigrams)
3. **Meta Features** – Star rating, helpful votes, verified purchase, review length
4. **ML Pipeline** – Three models trained and compared; best model selected
5. **Output** – Liked / Disliked prediction with confidence percentage

---

## ☁️ Deploy on Streamlit Cloud

1. Push project to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Add a startup command to train the model:
   - In `packages.txt` or via a `setup.sh`, run:  
     `python data/generate_data.py && python train_model.py`

---

## 🛠️ Tech Stack

- **Frontend** – Streamlit with custom CSS
- **ML** – Scikit-learn (Logistic Regression, Random Forest, GBM)
- **NLP** – TF-IDF Vectorization
- **Data** – Pandas, NumPy
