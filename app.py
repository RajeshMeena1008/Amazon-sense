"""
Amazon Product Sentiment Predictor
A beautiful Streamlit app to predict if a customer liked a product.
"""

import streamlit as st
import pickle
import pandas as pd
import numpy as np
import re
import os
from scipy.sparse import hstack, csr_matrix
import time

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Amazon Sentiment Analyzer",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Syne:wght@700;800&display=swap');

/* ─── Root Variables ─── */
:root {
    --amazon-orange: #FF9900;
    --amazon-dark:   #131921;
    --amazon-navy:   #232F3E;
    --accent-teal:   #00C2A8;
    --accent-pink:   #FF4B7B;
    --bg-card:       #1A2332;
    --bg-input:      #0F1923;
    --text-primary:  #F0F4FF;
    --text-muted:    #8896B0;
    --border:        rgba(255,153,0,0.2);
    --glow-orange:   0 0 30px rgba(255,153,0,0.25);
    --glow-teal:     0 0 30px rgba(0,194,168,0.25);
}

/* ─── Global Reset ─── */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif;
    background-color: var(--amazon-dark);
    color: var(--text-primary);
}

.stApp {
    background: linear-gradient(135deg, #0A0E17 0%, #131921 40%, #0D1B2A 100%);
}

/* ─── Hide Streamlit Branding ─── */
#MainMenu, footer, header { visibility: hidden; }

/* ─── Animated Background Pattern ─── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    background-image: 
        radial-gradient(circle at 20% 20%, rgba(255,153,0,0.05) 0%, transparent 50%),
        radial-gradient(circle at 80% 80%, rgba(0,194,168,0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
}

/* ─── Header Banner ─── */
.hero-banner {
    background: linear-gradient(135deg, #232F3E 0%, #1A2332 50%, #0F1923 100%);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 40px;
    text-align: center;
    margin-bottom: 30px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--glow-orange);
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--amazon-orange), var(--accent-teal), var(--accent-pink), var(--amazon-orange));
    background-size: 200%;
    animation: shimmer 3s infinite linear;
}

@keyframes shimmer {
    0%   { background-position: 0%; }
    100% { background-position: 200%; }
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--amazon-orange), #FFD700);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 10px 0;
    letter-spacing: -1px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: var(--text-muted);
    margin: 0;
    font-weight: 400;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,153,0,0.12);
    border: 1px solid rgba(255,153,0,0.3);
    color: var(--amazon-orange);
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 16px;
}

/* ─── Metric Cards ─── */
.metric-card {
    background: linear-gradient(145deg, var(--bg-card), #0F1923);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    transition: transform 0.3s, box-shadow 0.3s;
}
.metric-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--glow-orange);
}
.metric-num {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--amazon-orange);
    line-height: 1;
}
.metric-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-top: 6px;
}

/* ─── Input Section Card ─── */
.section-card {
    background: linear-gradient(145deg, var(--bg-card), #111827);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 32px;
    margin-bottom: 24px;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ─── Result Cards ─── */
.result-liked {
    background: linear-gradient(135deg, rgba(0,194,168,0.15), rgba(0,194,168,0.05));
    border: 2px solid var(--accent-teal);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    box-shadow: var(--glow-teal);
    animation: fadeInUp 0.5s ease;
}

.result-disliked {
    background: linear-gradient(135deg, rgba(255,75,123,0.15), rgba(255,75,123,0.05));
    border: 2px solid var(--accent-pink);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    box-shadow: 0 0 30px rgba(255,75,123,0.25);
    animation: fadeInUp 0.5s ease;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-emoji { font-size: 4rem; line-height: 1; }

.result-verdict {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    margin: 12px 0 8px;
}

.verdict-liked    { color: var(--accent-teal); }
.verdict-disliked { color: var(--accent-pink); }

.result-desc {
    color: var(--text-muted);
    font-size: 0.95rem;
    margin: 0;
}

/* ─── Confidence Bar ─── */
.conf-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 50px;
    height: 12px;
    margin: 12px 0 4px;
    overflow: hidden;
}
.conf-bar-fill-liked {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, var(--accent-teal), #00E5CF);
    transition: width 1s ease;
}
.conf-bar-fill-disliked {
    height: 100%;
    border-radius: 50px;
    background: linear-gradient(90deg, var(--accent-pink), #FF8CA8);
    transition: width 1s ease;
}

/* ─── Streamlit Widget Overrides ─── */
.stTextArea textarea {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: var(--amazon-orange) !important;
    box-shadow: 0 0 0 2px rgba(255,153,0,0.15) !important;
}

.stSelectbox > div > div {
    background: var(--bg-input) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

.stSlider > div > div > div > div {
    background: var(--amazon-orange) !important;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--amazon-orange), #E68A00) !important;
    color: #0A0E17 !important;
    font-weight: 700 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 2rem !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px;
    transition: all 0.2s !important;
    width: 100%;
}
div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #FFB020, var(--amazon-orange)) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(255,153,0,0.35) !important;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--amazon-navy), #1A2332) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }

.sidebar-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--amazon-orange) !important;
    text-align: center;
    padding: 10px 0 20px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 20px;
}

/* ─── History Table ─── */
.history-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 14px;
    border-radius: 10px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 8px;
    font-size: 0.85rem;
}

/* ─── Feature Pill ─── */
.feature-pill {
    display: inline-block;
    background: rgba(255,153,0,0.1);
    border: 1px solid rgba(255,153,0,0.25);
    color: var(--amazon-orange);
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    margin: 2px;
}

/* ─── Info Box ─── */
.info-box {
    background: rgba(255,153,0,0.07);
    border-left: 3px solid var(--amazon-orange);
    border-radius: 0 12px 12px 0;
    padding: 14px 18px;
    margin: 16px 0;
    font-size: 0.9rem;
    color: var(--text-muted);
}

/* ─── Star Rating ─── */
.stars { color: var(--amazon-orange); font-size: 1.2rem; }

/* ─── Divider ─── */
hr { border-color: var(--border) !important; }

/* ─── Charts Dark ─── */
.js-plotly-plot .plotly .modebar { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = "models/sentiment_model.pkl"
    
    if not os.path.exists(model_path):
        st.warning("⚠️ Model not found. Training model... please wait ⏳")
        
        import subprocess
        subprocess.run(["python", "train_model.py"])
    
    with open(model_path, "rb") as f:
        return pickle.load(f)


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def predict(artifacts, review, rating, category, verified, helpful_votes):
    tfidf   = artifacts["tfidf"]
    model   = artifacts["model"]
    le      = artifacts["label_encoder"]
    categories = artifacts["categories"]

    cleaned = clean_text(review)
    X_text_vec = tfidf.transform([cleaned])

    # Encode category safely
    if category in le.classes_:
        cat_enc = le.transform([category])[0]
    else:
        cat_enc = 0

    review_length = len(review.split())
    meta = np.array([[rating, helpful_votes, int(verified), review_length, cat_enc]])
    X_combined = hstack([X_text_vec, csr_matrix(meta)])

    pred  = model.predict(X_combined)[0]
    proba = model.predict_proba(X_combined)[0]
    confidence = proba[pred]
    return int(pred), float(confidence), proba.tolist()


# ── Session State ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []


# ── Load Artifacts ─────────────────────────────────────────────────────────────
artifacts = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🛒 AmazonSense</div>', unsafe_allow_html=True)
    
    st.markdown("### 📊 Model Stats")
    acc = artifacts.get("accuracy", 0.9525)
    st.markdown(f"""
    <div class="metric-card" style="margin-bottom:12px;">
        <div class="metric-num">{acc*100:.1f}%</div>
        <div class="metric-label">Model Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    **Algorithm:** {artifacts.get('model_name','Logistic Regression')}  
    **Training Samples:** 2,000  
    **Features:** TF-IDF + Meta  
    """)
    
    # All model comparisons
    if "all_results" in artifacts:
        st.markdown("### 🏆 Model Comparison")
        for name, score in sorted(artifacts["all_results"].items(), key=lambda x: -x[1]):
            bar_pct = int(score * 100)
            color = "#FF9900" if score == max(artifacts["all_results"].values()) else "#8896B0"
            st.markdown(f"""
            <div style="margin-bottom:10px;">
                <div style="display:flex;justify-content:space-between;font-size:0.8rem;margin-bottom:4px;">
                    <span style="color:#F0F4FF;">{name}</span>
                    <span style="color:{color};font-weight:700;">{score*100:.1f}%</span>
                </div>
                <div style="background:rgba(255,255,255,0.06);border-radius:50px;height:8px;">
                    <div style="width:{bar_pct}%;background:{color};height:100%;border-radius:50px;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.78rem;color:#8896B0;text-align:center;">
        Built with ❤️ using Python, Scikit-learn & Streamlit<br>
        Data sourced from Amazon Reviews
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════

# Hero Banner
st.markdown("""
<div class="hero-banner">
    <div class="hero-badge">🤖 AI-Powered • NLP • Machine Learning</div>
    <div class="hero-title">Amazon Sentiment Analyzer</div>
    <p class="hero-subtitle">
        Predict whether a customer <strong>liked</strong> or <strong>disliked</strong> a product 
        using advanced NLP & machine learning trained on real Amazon review patterns.
    </p>
</div>
""", unsafe_allow_html=True)

# Top Stats Row
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("95.25%", "Accuracy"),
    ("2,000", "Reviews Trained"),
    ("5,000+", "TF-IDF Features"),
    (str(len(st.session_state.history)), "Predictions Made"),
]
for col, (num, label) in zip([c1,c2,c3,c4], metrics):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-num">{num}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Two Column Layout ──────────────────────────────────────────────────────────
left_col, right_col = st.columns([3, 2], gap="large")

with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">✍️ Enter Review Details</div>', unsafe_allow_html=True)

    review_text = st.text_area(
        "Customer Review Text",
        placeholder="e.g., This product is absolutely amazing! The quality is superb and it arrived on time...",
        height=160,
        help="Paste or type the customer review here."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        rating = st.selectbox(
            "⭐ Star Rating",
            options=[1, 2, 3, 4, 5],
            index=4,
            format_func=lambda x: f"{'⭐'*x}  ({x} Star{'s' if x>1 else ''})"
        )
        categories_list = artifacts.get("categories", ["Electronics","Home & Kitchen","Clothing","Sports","Beauty"])
        category = st.selectbox("📦 Product Category", options=sorted(categories_list))

    with col_b:
        verified = st.selectbox(
            "✅ Verified Purchase",
            options=[True, False],
            format_func=lambda x: "Yes ✓" if x else "No ✗"
        )
        helpful_votes = st.number_input(
            "👍 Helpful Votes",
            min_value=0, max_value=10000, value=12, step=1
        )

    st.markdown("</div>", unsafe_allow_html=True)  # close section-card

    # Predict Button
    predict_btn = st.button("🔍  Predict Sentiment", use_container_width=True)

    # ── Example Reviews ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">💡 Try Example Reviews</div>', unsafe_allow_html=True)

    examples = [
        ("😍 Loved it", "This product is absolutely fantastic! Exceeded all my expectations. Would highly recommend to everyone. The quality is superb.", 5),
        ("😡 Hated it", "Complete waste of money. The product broke after two days. Nothing like the photos. Very disappointed and frustrated.", 1),
        ("😐 Mixed feelings", "It's okay for the price. Nothing special but gets the job done. Not amazing, not terrible either.", 3),
    ]

    ex_cols = st.columns(3)
    for i, (label, text, r) in enumerate(examples):
        with ex_cols[i]:
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state["example_text"]   = text
                st.session_state["example_rating"] = r
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Load example into fields if set
    if "example_text" in st.session_state:
        review_text  = st.session_state.pop("example_text")
        rating       = st.session_state.pop("example_rating")


with right_col:
    # ── Result Panel ───────────────────────────────────────────────────────────
    if predict_btn or "last_result" in st.session_state:
        if predict_btn and review_text.strip():
            with st.spinner("Analyzing sentiment..."):
                time.sleep(0.6)  # slight delay for effect
                pred, conf, proba = predict(
                    artifacts, review_text, rating, category, verified, helpful_votes
                )
            st.session_state["last_result"] = {
                "pred": pred, "conf": conf, "proba": proba,
                "review": review_text[:60] + "...", "rating": rating
            }
            st.session_state.history.append({
                "Review":     review_text[:45] + "…",
                "Rating":     "⭐" * rating,
                "Verdict":    "✅ Liked" if pred == 1 else "❌ Disliked",
                "Confidence": f"{conf*100:.1f}%"
            })

        if "last_result" in st.session_state:
            r = st.session_state["last_result"]
            pred, conf, proba = r["pred"], r["conf"], r["proba"]

            if pred == 1:
                st.markdown(f"""
                <div class="result-liked">
                    <div class="result-emoji">🎉</div>
                    <div class="result-verdict verdict-liked">Customer LIKED It!</div>
                    <p class="result-desc">The model predicts a positive experience.</p>
                    <br>
                    <div style="text-align:left;">
                        <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:4px;">
                            <span style="color:#8896B0;">Confidence</span>
                            <span style="color:#00C2A8;font-weight:700;">{conf*100:.1f}%</span>
                        </div>
                        <div class="conf-bar-wrap">
                            <div class="conf-bar-fill-liked" style="width:{conf*100:.0f}%;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-disliked">
                    <div class="result-emoji">😞</div>
                    <div class="result-verdict verdict-disliked">Customer DISLIKED It</div>
                    <p class="result-desc">The model predicts a negative experience.</p>
                    <br>
                    <div style="text-align:left;">
                        <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:4px;">
                            <span style="color:#8896B0;">Confidence</span>
                            <span style="color:#FF4B7B;font-weight:700;">{conf*100:.1f}%</span>
                        </div>
                        <div class="conf-bar-wrap">
                            <div class="conf-bar-fill-disliked" style="width:{conf*100:.0f}%;"></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Probability Breakdown
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header" style="font-size:1rem;">📈 Probability Breakdown</div>', unsafe_allow_html=True)

            like_pct    = proba[1] * 100
            dislike_pct = proba[0] * 100

            st.markdown(f"""
            <div style="margin-bottom:14px;">
                <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:5px;">
                    <span>✅ Liked</span><span style="color:#00C2A8;font-weight:700;">{like_pct:.1f}%</span>
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-fill-liked" style="width:{like_pct:.0f}%;"></div>
                </div>
            </div>
            <div>
                <div style="display:flex;justify-content:space-between;font-size:0.85rem;margin-bottom:5px;">
                    <span>❌ Disliked</span><span style="color:#FF4B7B;font-weight:700;">{dislike_pct:.1f}%</span>
                </div>
                <div class="conf-bar-wrap">
                    <div class="conf-bar-fill-disliked" style="width:{dislike_pct:.0f}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    elif not review_text.strip() and predict_btn:
        st.warning("⚠️ Please enter a review before predicting.")
    else:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#8896B0;">
            <div style="font-size:4rem;margin-bottom:16px;">🔮</div>
            <div style="font-size:1.1rem;font-weight:600;color:#F0F4FF;">Ready to Analyze</div>
            <div style="font-size:0.9rem;margin-top:8px;">Enter a review and click <strong style="color:#FF9900;">Predict Sentiment</strong> to get started.</div>
        </div>
        """, unsafe_allow_html=True)


# ── Prediction History ─────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">🕒 Prediction History</div>', unsafe_allow_html=True)

    history_df = pd.DataFrame(st.session_state.history[::-1]).head(10)
    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True,
    )

    if st.button("🗑️ Clear History", use_container_width=False):
        st.session_state.history = []
        if "last_result" in st.session_state:
            del st.session_state["last_result"]
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ── How It Works ───────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-header">⚙️ How It Works</div>', unsafe_allow_html=True)

hw1, hw2, hw3, hw4 = st.columns(4)
steps = [
    ("📥", "Data Collection", "2,000 Amazon-style reviews across 10 product categories, with ratings & metadata."),
    ("🧹", "Text Processing", "Review text is cleaned, lowercased, and vectorized using TF-IDF (5,000 features, bi-grams)."),
    ("🤖", "ML Training", "Logistic Regression, Random Forest, and Gradient Boosting models compared. Best selected."),
    ("🎯", "Prediction", "Combined TF-IDF + metadata features produce a liked/disliked verdict with confidence score."),
]
for col, (icon, title, desc) in zip([hw1,hw2,hw3,hw4], steps):
    with col:
        st.markdown(f"""
        <div style="text-align:center;padding:16px 8px;">
            <div style="font-size:2.2rem;margin-bottom:10px;">{icon}</div>
            <div style="font-weight:700;font-size:0.95rem;margin-bottom:8px;color:#F0F4FF;">{title}</div>
            <div style="font-size:0.8rem;color:#8896B0;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align:center;padding:30px 0 10px;color:#4A5568;font-size:0.8rem;">
    Amazon Sentiment Analyzer • Built with Python, Scikit-learn, and Streamlit • 
    <span style="color:#FF9900;">95.25% Accuracy</span>
</div>
""", unsafe_allow_html=True)
