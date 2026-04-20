"""
========================================================================
Sentiment Analysis of Product Reviews Using TF-IDF and ML Classifiers
========================================================================
Author  : Production ML Pipeline
Version : 1.0.0
------------------------------------------------------------------------
Pipeline:
  1. Data Loading & Labeling
  2. Text Preprocessing
  3. Feature Engineering (TF-IDF + Linguistic)
  4. Feature Selection (Chi2 + Mutual Information)
  5. Model Training & Evaluation
  6. Model Comparison & Explainability
  7. Visualization
  8. Model Saving
========================================================================
"""

# ─────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────
import os
import re
import time
import pickle
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
RANDOM_STATE   = 42
TEST_SIZE      = 0.2
TOP_K_CHI2     = 500
TOP_K_MI       = 300
TFIDF_MAX_FEAT = 15000
PLOTS_DIR      = "plots"
MODELS_DIR     = "saved_models"

# Built-in English stopwords (no NLTK download required)
STOPWORDS = {
    "a","about","above","after","again","against","all","am","an","and","any","are",
    "aren't","as","at","be","because","been","before","being","below","between","both",
    "but","by","can't","cannot","could","couldn't","did","didn't","do","does","doesn't",
    "doing","don't","down","during","each","few","for","from","further","get","got","had",
    "hadn't","has","hasn't","have","haven't","having","he","he'd","he'll","he's","her",
    "here","here's","hers","herself","him","himself","his","how","how's","i","i'd","i'll",
    "i'm","i've","if","in","into","is","isn't","it","it's","its","itself","let's","me",
    "more","most","mustn't","my","myself","no","nor","not","of","off","on","once","only",
    "or","other","ought","our","ours","ourselves","out","over","own","same","shan't","she",
    "she'd","she'll","she's","should","shouldn't","so","some","such","than","that","that's",
    "the","their","theirs","them","themselves","then","there","there's","these","they",
    "they'd","they'll","they're","they've","this","those","through","to","too","under",
    "until","up","very","was","wasn't","we","we'd","we'll","we're","we've","were",
    "weren't","what","what's","when","when's","where","where's","which","while","who",
    "who's","whom","why","why's","will","with","won't","would","wouldn't","you","you'd",
    "you'll","you're","you've","your","yours","yourself","yourselves","just","also",
    "however","therefore","furthermore","thus","hence","indeed","still","yet","though",
    "although","since","while","unless","whether","either","neither","both","each",
    "every","much","many","more","most","such","no","nor","not","only","own","same","so"
}

# Negation words for linguistic feature
NEGATION_WORDS = {
    "not","no","never","none","nobody","nothing","neither","nor","nowhere",
    "hardly","barely","scarcely","doesn't","didn't","don't","won't","wouldn't",
    "couldn't","shouldn't","isn't","aren't","wasn't","weren't","haven't","hasn't",
    "hadn't","can't","cannot","shan't","mustn't","needn't","daren't","without"
}

SENTIMENT_MAP = {1: "Negative", 2: "Negative", 3: "Neutral", 4: "Positive", 5: "Positive"}
SENTIMENT_ORDER = ["Negative", "Neutral", "Positive"]
PALETTE = {"Negative": "#E74C3C", "Neutral": "#F39C12", "Positive": "#27AE60"}


# ═══════════════════════════════════════════════════════════
# SECTION 1 – DATA LOADING
# ═══════════════════════════════════════════════════════════

def load_dataset(filepath: str) -> pd.DataFrame:

    print(f"\n{'='*60}")
    print("SECTION 1: DATA LOADING")
    print(f"{'='*60}")

    print(f"  ► Loading file: {filepath}")

    # Read CSV
    try:
        df = pd.read_csv(
            filepath,
            on_bad_lines="skip",
            encoding="utf-8",
            engine="python"
)
    except UnicodeDecodeError:
        df = pd.read_csv(
            filepath,
            on_bad_lines="skip",
            encoding="latin-1",
            engine="python"
)

    print(f"  ► Raw shape: {df.shape}")
    print(f"  ► Columns  : {list(df.columns)}")

    # Detect columns
    text_col = "Review Text"
    rating_col = "Rating"

    df = df[[text_col, rating_col]].copy()
    df.columns = ["review_text", "rating"]

    # 🔥 FIXED rating extraction
    df["rating"] = df["rating"].astype(str).str.extract(r'(\d)').astype(float)

    # Clean
    df = df.dropna(subset=["review_text", "rating"])
    df = df[df["rating"].between(1, 5)]
    df["rating"] = df["rating"].astype(int)

    # Safety check
    if len(df) == 0:
        raise ValueError("All rows removed. Check rating format.")

    # Map sentiment
    df["sentiment"] = df["rating"].map(SENTIMENT_MAP)

    print(f"  ► Clean shape : {df.shape}")
    print(f"\n  Sentiment distribution:")

    counts = df["sentiment"].value_counts()

    for s in SENTIMENT_ORDER:
        n = counts.get(s, 0)
        pct = 100 * n / len(df)
        print(f"    {s:10s}: {n:6,d}  ({pct:.1f}%)")

    return df.reset_index(drop=True)

# ═══════════════════════════════════════════════════════════
# SECTION 2 – TEXT PREPROCESSING
# ═══════════════════════════════════════════════════════════

def clean_text(text: str) -> str:
    """
    Clean a single review string:
      1. Lowercase
      2. Strip HTML tags
      3. Remove special characters / numbers
      4. Collapse whitespace
      5. Remove stopwords
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove special characters and digits, keep letters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords (single-pass via join/split)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]

    return " ".join(tokens)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to the dataset and add a 'cleaned_text' column.
    """
    print(f"\n{'='*60}")
    print("SECTION 2: TEXT PREPROCESSING")
    print(f"{'='*60}")

    before = len(df)
    df = df.dropna(subset=["review_text"]).copy()
    df = df[df["review_text"].str.strip() != ""]
    print(f"  ► Rows after null/empty drop : {len(df):,}  (removed {before-len(df):,})")

    print("  ► Cleaning text (lowercase, HTML removal, special chars, stopwords)…")
    df["cleaned_text"] = df["review_text"].apply(clean_text)

    # Drop rows where cleaning left empty strings
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    print(f"  ► Final rows after cleaning  : {len(df):,}")

    # Sample stats
    avg_len = df["cleaned_text"].str.split().str.len().mean()
    print(f"  ► Avg tokens per review      : {avg_len:.1f}")

    return df


# ═══════════════════════════════════════════════════════════
# SECTION 3 – FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════

def extract_linguistic_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute 4 hand-crafted linguistic features per review:
      1. sentence_length      – number of words in cleaned text
      2. punctuation_density  – punctuation chars / total chars (raw text)
      3. avg_word_length      – mean character length of words
      4. has_negation         – 1 if any negation word present, else 0
    """
    feats = pd.DataFrame()

    # 1. Sentence length (word count of cleaned text)
    feats["sentence_length"] = df["cleaned_text"].str.split().str.len().fillna(0)

    # 2. Punctuation density (raw text)
    def punct_density(s):
        if not isinstance(s, str) or len(s) == 0:
            return 0.0
        punct = sum(1 for c in s if c in "!?,.:;\"'()[]{}…-")
        return punct / len(s)

    feats["punctuation_density"] = df["review_text"].apply(punct_density)

    # 3. Average word length (cleaned text)
    def avg_word_len(s):
        words = s.split() if isinstance(s, str) else []
        if not words:
            return 0.0
        return np.mean([len(w) for w in words])

    feats["avg_word_length"] = df["cleaned_text"].apply(avg_word_len)

    # 4. Presence of negation words (raw text)
    def has_negation(s):
        if not isinstance(s, str):
            return 0
        tokens = set(re.sub(r"[^a-z\s']", " ", s.lower()).split())
        return int(bool(tokens & NEGATION_WORDS))

    feats["has_negation"] = df["review_text"].apply(has_negation)

    return feats.values.astype(np.float32)


def build_features(df: pd.DataFrame):
    """
    Build combined feature matrix:
      - TF-IDF with unigrams + bigrams (sparse)
      - Linguistic features (dense → sparse)

    Returns
    -------
    X_combined : scipy sparse matrix
    tfidf      : fitted TfidfVectorizer
    """
    print(f"\n{'='*60}")
    print("SECTION 3: FEATURE ENGINEERING")
    print(f"{'='*60}")

    # ── TF-IDF ───────────────────────────────────────────────
    print(f"  ► Fitting TF-IDF (max_features={TFIDF_MAX_FEAT}, "
          f"ngram_range=(1,2))…")
    tfidf = TfidfVectorizer(
        max_features=TFIDF_MAX_FEAT,
        ngram_range=(1, 2),
        sublinear_tf=True,          # log(1+tf)
        min_df=3,
        max_df=0.95,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-z]{2,}\b"
    )
    X_tfidf = tfidf.fit_transform(df["cleaned_text"])
    print(f"  ► TF-IDF matrix shape  : {X_tfidf.shape}")

    # ── Linguistic features ──────────────────────────────────
    print("  ► Extracting linguistic features…")
    X_ling = extract_linguistic_features(df)
    X_ling_sparse = csr_matrix(X_ling)
    print(f"  ► Linguistic feats shape: {X_ling.shape}")

    # ── Combine ──────────────────────────────────────────────
    X_combined = hstack([X_tfidf, X_ling_sparse])
    print(f"  ► Combined feature shape: {X_combined.shape}")

    return X_combined, tfidf


# ═══════════════════════════════════════════════════════════
# SECTION 4 – FEATURE SELECTION
# ═══════════════════════════════════════════════════════════

def select_features(X, y, method: str = "chi2", k: int = TOP_K_CHI2):
    """
    Apply univariate feature selection.

    Parameters
    ----------
    X      : sparse feature matrix
    y      : encoded label array
    method : 'chi2' or 'mutual_info'
    k      : number of top features to keep

    Returns
    -------
    X_selected : reduced sparse matrix
    selector   : fitted SelectKBest object
    """
    print(f"\n  ► Feature selection — method='{method}', k={k}")

    score_func = chi2 if method == "chi2" else mutual_info_classif
    selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    print(f"    Selected shape: {X_selected.shape}")
    return X_selected, selector


# ═══════════════════════════════════════════════════════════
# SECTION 5 – MODEL TRAINING & EVALUATION
# ═══════════════════════════════════════════════════════════

def get_models() -> dict:
    """Return a dictionary of model name → estimator."""
    return {
        "Logistic Regression": LogisticRegression(
    C=1.0,
    max_iter=1000,
    solver="lbfgs",
    class_weight="balanced",
    random_state=RANDOM_STATE
),
        "Multinomial NB": MultinomialNB(alpha=0.5),
        "Linear SVM": CalibratedClassifierCV(
    LinearSVC(
        C=1.0, max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),
    cv=3
),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.1, max_depth=4,
            subsample=0.8, random_state=RANDOM_STATE
        ),
    }


def evaluate_model(model, X_train, X_test, y_train, y_test,
                   label_names: list) -> dict:
    """
    Train a model, evaluate on test set, and return a metrics dict.
    """
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy"  : accuracy_score(y_test, y_pred),
        "precision" : precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall"    : recall_score(y_test, y_pred, average="macro", zero_division=0),
        "macro_f1"  : f1_score(y_test, y_pred, average="macro", zero_division=0),
        "train_time": train_time,
        "report"    : classification_report(y_test, y_pred,
                                            target_names=label_names,
                                            zero_division=0),
        "conf_matrix": confusion_matrix(y_test, y_pred),
        "y_pred"    : y_pred,
    }
    return metrics


def train_and_evaluate(X_train, X_test, y_train, y_test,
                       label_names: list) -> tuple[dict, dict]:
    """
    Train all models and collect evaluation results.

    Returns
    -------
    results  : dict  { model_name → metrics_dict }
    models   : dict  { model_name → fitted estimator }
    """
    print(f"\n{'='*60}")
    print("SECTION 5: MODEL TRAINING & EVALUATION")
    print(f"{'='*60}")

    models_def = get_models()
    results = {}
    fitted_models = {}

    for name, model in models_def.items():
        print(f"\n  ▷ Training: {name} …")
        metrics = evaluate_model(model, X_train, X_test, y_train, y_test, label_names)
        results[name] = metrics
        fitted_models[name] = model

        print(f"    Accuracy  : {metrics['accuracy']:.4f}")
        print(f"    Precision : {metrics['precision']:.4f}")
        print(f"    Recall    : {metrics['recall']:.4f}")
        print(f"    Macro F1  : {metrics['macro_f1']:.4f}")
        print(f"    Train time: {metrics['train_time']:.2f}s")
        print()
        print("  Classification Report:")
        for line in metrics["report"].splitlines():
            print(f"    {line}")

    return results, fitted_models


# ═══════════════════════════════════════════════════════════
# SECTION 6 – MODEL COMPARISON
# ═══════════════════════════════════════════════════════════

def compare_models(results: dict) -> pd.DataFrame:
    """
    Build a comparison DataFrame and identify best model by Macro F1.
    """
    print(f"\n{'='*60}")
    print("SECTION 6: MODEL COMPARISON")
    print(f"{'='*60}")

    rows = []
    for name, m in results.items():
        rows.append({
            "Model"     : name,
            "Accuracy"  : round(m["accuracy"], 4),
            "Precision" : round(m["precision"], 4),
            "Recall"    : round(m["recall"], 4),
            "Macro F1"  : round(m["macro_f1"], 4),
            "Train(s)"  : round(m["train_time"], 2),
        })

    cmp_df = pd.DataFrame(rows).sort_values("Macro F1", ascending=False).reset_index(drop=True)
    best   = cmp_df.iloc[0]["Model"]

    print(f"\n  {'─'*65}")
    print(cmp_df.to_string(index=False))
    print(f"  {'─'*65}")
    print(f"\n  ★  Best model by Macro F1 → {best}  "
          f"(F1 = {cmp_df.iloc[0]['Macro F1']:.4f})")

    return cmp_df, best


# ═══════════════════════════════════════════════════════════
# SECTION 7 – MODEL EXPLAINABILITY
# ═══════════════════════════════════════════════════════════

def get_feature_names(tfidf: TfidfVectorizer, selector, n_ling: int = 4) -> list:
    """Reconstruct feature names after SelectKBest."""
    ling_names = ["sentence_length", "punctuation_density",
                  "avg_word_length", "has_negation"]
    all_names  = list(tfidf.get_feature_names_out()) + ling_names[:n_ling]
    mask       = selector.get_support()
    return [all_names[i] for i, keep in enumerate(mask) if keep]


def explain_model(model, model_name: str, feature_names: list,
                  label_names: list, top_n: int = 20):
    """
    Extract top discriminating features for each sentiment class.
    Works for Logistic Regression and Linear SVM (coefficient-based).
    """
    print(f"\n{'='*60}")
    print("SECTION 7: MODEL EXPLAINABILITY")
    print(f"{'='*60}")
    print(f"  ► Model: {model_name}")

    # Unwrap CalibratedClassifierCV
    base = model.estimator if hasattr(model, "estimator") else model

    if not hasattr(base, "coef_"):
        print("  ► Skipping (no coef_ attribute — Gradient Boosting / NB)")
        return {}

    coef = base.coef_           # shape: (n_classes, n_features)
    top_features = {}

    for i, cls in enumerate(label_names):
        if coef.shape[0] == 1:  # binary OvR
            c = coef[0] if i == 1 else -coef[0]
        else:
            c = coef[i]

        top_idx     = np.argsort(c)[::-1][:top_n]
        bottom_idx  = np.argsort(c)[:top_n]

        top_features[cls] = {
            "positive": [(feature_names[j], c[j]) for j in top_idx],
            "negative": [(feature_names[j], c[j]) for j in bottom_idx],
        }
        print(f"\n  Class: {cls}")
        print(f"    Top {top_n} positive features:")
        for feat, score in top_features[cls]["positive"][:10]:
            print(f"      {feat:<35s}  {score:+.4f}")

    return top_features


# ═══════════════════════════════════════════════════════════
# SECTION 8 – VISUALISATIONS
# ═══════════════════════════════════════════════════════════

def save_plot(fig, name: str):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ► Saved: {path}")


def plot_sentiment_distribution(df: pd.DataFrame):
    """Bar + pie chart of sentiment distribution."""
    counts = df["sentiment"].value_counts().reindex(SENTIMENT_ORDER)
    colors = [PALETTE[s] for s in SENTIMENT_ORDER]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sentiment Distribution", fontsize=16, fontweight="bold")

    # Bar chart
    axes[0].bar(SENTIMENT_ORDER, counts.values, color=colors, edgecolor="white",
                linewidth=1.2, width=0.55)
    axes[0].set_xlabel("Sentiment", fontsize=12)
    axes[0].set_ylabel("Count", fontsize=12)
    axes[0].set_title("Review Counts by Sentiment")
    for i, (v, c) in enumerate(zip(counts.values, colors)):
        axes[0].text(i, v + counts.max() * 0.01, f"{v:,}", ha="center",
                     fontsize=11, fontweight="bold", color=c)

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=SENTIMENT_ORDER, colors=colors,
        autopct="%1.1f%%", startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
    axes[1].set_title("Sentiment Proportions")

    plt.tight_layout()
    save_plot(fig, "01_sentiment_distribution.png")


def plot_review_length(df: pd.DataFrame):
    """Histogram of review word counts per sentiment class."""
    df = df.copy()
    df["word_count"] = df["cleaned_text"].str.split().str.len()

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle("Review Length Distribution by Sentiment", fontsize=16, fontweight="bold")

    for sent in SENTIMENT_ORDER:
        subset = df.loc[df["sentiment"] == sent, "word_count"]
        ax.hist(subset, bins=50, alpha=0.55, label=sent,
                color=PALETTE[sent], edgecolor="none")

    ax.set_xlabel("Word Count (cleaned)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 200)
    plt.tight_layout()
    save_plot(fig, "02_review_length_distribution.png")


def plot_model_comparison(cmp_df: pd.DataFrame):
    """Grouped bar chart comparing models across metrics."""
    metrics   = ["Accuracy", "Precision", "Recall", "Macro F1"]
    x         = np.arange(len(cmp_df))
    bar_width = 0.18
    colors    = ["#3498DB", "#E74C3C", "#27AE60", "#9B59B6"]

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.suptitle("Model Comparison", fontsize=16, fontweight="bold")

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        offset = (i - 1.5) * bar_width
        bars   = ax.bar(x + offset, cmp_df[metric], bar_width,
                        label=metric, color=color, alpha=0.88, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.003,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(cmp_df["Model"], fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=12)
    ax.legend(fontsize=10, loc="upper right")
    ax.axhline(y=1.0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    save_plot(fig, "03_model_comparison.png")


def plot_confusion_matrices(results: dict, label_names: list):
    """2×2 grid of confusion matrices for all models."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Confusion Matrices", fontsize=16, fontweight="bold")
    axes = axes.flatten()

    for idx, (name, m) in enumerate(results.items()):
        cm   = m["conf_matrix"]
        cm_n = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)   # row-normalise

        sns.heatmap(cm_n, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names,
                    ax=axes[idx], linewidths=0.5, cbar=False,
                    annot_kws={"size": 12, "weight": "bold"})
        axes[idx].set_title(name, fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Predicted", fontsize=10)
        axes[idx].set_ylabel("Actual", fontsize=10)

    plt.tight_layout()
    save_plot(fig, "04_confusion_matrices.png")


def plot_top_words(df: pd.DataFrame, n: int = 15):
    """Top N most frequent words per sentiment class."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    fig.suptitle(f"Top {n} Words per Sentiment", fontsize=16, fontweight="bold")

    for ax, sent in zip(axes, SENTIMENT_ORDER):
        corpus = " ".join(df.loc[df["sentiment"] == sent, "cleaned_text"])
        freq   = pd.Series(corpus.split()).value_counts().head(n)

        bars = ax.barh(freq.index[::-1], freq.values[::-1],
                       color=PALETTE[sent], alpha=0.85, edgecolor="white")
        ax.set_title(sent, fontsize=13, fontweight="bold", color=PALETTE[sent])
        ax.set_xlabel("Frequency", fontsize=10)
        ax.tick_params(axis="y", labelsize=9)

        for bar in bars:
            ax.text(bar.get_width() + freq.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{int(bar.get_width()):,}", va="center", fontsize=8)

    plt.tight_layout()
    save_plot(fig, "05_top_words_per_sentiment.png")


def plot_feature_importance(top_features: dict, model_name: str):
    """Visualise top features per sentiment from a linear model."""
    if not top_features:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(f"Top Discriminating Features — {model_name}",
                 fontsize=15, fontweight="bold")

    for ax, sent in zip(axes, SENTIMENT_ORDER):
        if sent not in top_features:
            continue
        feats  = top_features[sent]["positive"][:15]
        names  = [f for f, _ in feats]
        scores = [s for _, s in feats]

        colors_bar = [PALETTE[sent] if s > 0 else "#95A5A6" for s in scores]
        ax.barh(names[::-1], scores[::-1], color=colors_bar[::-1],
                edgecolor="white", alpha=0.85)
        ax.set_title(f"{sent}", fontsize=12, fontweight="bold", color=PALETTE[sent])
        ax.set_xlabel("Coefficient", fontsize=10)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

    plt.tight_layout()
    save_plot(fig, "06_feature_importance.png")


# ═══════════════════════════════════════════════════════════
# SECTION 9 – MODEL SAVING
# ═══════════════════════════════════════════════════════════

def save_artifacts(best_model_name: str, fitted_models: dict,
                   tfidf: TfidfVectorizer, selector,
                   label_encoder: LabelEncoder, cmp_df: pd.DataFrame):
    """
    Persist all artifacts needed for inference / Streamlit integration:
      - best_model.pkl
      - tfidf_vectorizer.pkl
      - feature_selector.pkl
      - label_encoder.pkl
      - metadata.json
    """
    print(f"\n{'='*60}")
    print("SECTION 9: MODEL SAVING")
    print(f"{'='*60}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    objects = {
        "best_model.pkl"        : fitted_models[best_model_name],
        "tfidf_vectorizer.pkl"  : tfidf,
        "feature_selector.pkl"  : selector,
        "label_encoder.pkl"     : label_encoder,
    }

    for filename, obj in objects.items():
        path = os.path.join(MODELS_DIR, filename)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  ► Saved: {path}")

    # Metadata JSON
    best_row = cmp_df[cmp_df["Model"] == best_model_name].iloc[0]
    metadata = {
        "best_model"     : best_model_name,
        "accuracy"       : float(best_row["Accuracy"]),
        "precision"      : float(best_row["Precision"]),
        "recall"         : float(best_row["Recall"]),
        "macro_f1"       : float(best_row["Macro F1"]),
        "label_classes"  : list(label_encoder.classes_),
        "tfidf_vocab_size": len(tfidf.vocabulary_),
        "feature_selection": {
            "method": "chi2",
            "k"     : TOP_K_CHI2
        },
        "sentiment_map"  : SENTIMENT_MAP,
    }
    meta_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ► Saved: {meta_path}")

    print(f"\n  All artifacts saved to '{MODELS_DIR}/'")
    print("  Ready for Streamlit integration — see streamlit_app.py")


# ═══════════════════════════════════════════════════════════
# SECTION 10 – STREAMLIT-READY INFERENCE HELPER
# ═══════════════════════════════════════════════════════════

def load_inference_pipeline(models_dir: str = MODELS_DIR) -> dict:
    """
    Load saved artifacts for inference.
    Use this in a Streamlit app (see streamlit_app.py).

    Returns
    -------
    dict with keys: model, tfidf, selector, label_encoder
    """
    artifacts = {}
    for key, fname in [
        ("model",          "best_model.pkl"),
        ("tfidf",          "tfidf_vectorizer.pkl"),
        ("selector",       "feature_selector.pkl"),
        ("label_encoder",  "label_encoder.pkl"),
    ]:
        path = os.path.join(models_dir, fname)
        with open(path, "rb") as f:
            artifacts[key] = pickle.load(f)
    return artifacts


def predict_sentiment(text: str, artifacts: dict) -> dict:
    """
    Predict sentiment for a single raw review string.

    Parameters
    ----------
    text      : raw review text
    artifacts : dict from load_inference_pipeline()

    Returns
    -------
    dict with 'sentiment', 'probabilities'
    """
    cleaned = clean_text(text)
    df_tmp  = pd.DataFrame({"review_text": [text], "cleaned_text": [cleaned]})

    X_tfidf = artifacts["tfidf"].transform([cleaned])
    X_ling  = extract_linguistic_features(df_tmp)
    X_ling_sp = csr_matrix(X_ling)
    X_combined = hstack([X_tfidf, X_ling_sp])
    X_selected = artifacts["selector"].transform(X_combined)

    model   = artifacts["model"]
    pred    = model.predict(X_selected)[0]

    # Safe numeric → string label mapping (avoids label_encoder.inverse_transform issues)
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    label = label_map.get(int(pred), str(pred))

    probs = {}
    if hasattr(model, "predict_proba"):
        prob_arr = model.predict_proba(X_selected)[0]
        classes  = artifacts["label_encoder"].classes_
        for cls, p in zip(classes, prob_arr):
            # Map numeric class to string label; fall back to str(cls) if unknown
            str_label = str(cls)
            probs[str_label] = round(float(p), 4)

    return {"sentiment": label, "probabilities": probs}


# ═══════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════

def run_pipeline(csv_path: str): 
    """
    Execute the full sentiment analysis pipeline.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing reviews.
    """
    t_start = time.time()

    print("\n" + "█" * 60)
    print("  SENTIMENT ANALYSIS PIPELINE — PRODUCT REVIEWS")
    print("█" * 60)

    # 1. Load Data
    df = load_dataset(csv_path)

    # 2. Preprocess
    df = preprocess(df)

    # 3. Feature Engineering
    X, tfidf = build_features(df)

    # Encode labels
    le = LabelEncoder()
    le.fit(SENTIMENT_ORDER)
    y  = le.transform(df["sentiment"])

    # 4. Feature Selection (Chi2)
    print(f"\n{'='*60}")
    print("SECTION 4: FEATURE SELECTION")
    print(f"{'='*60}")
    X_chi2, sel_chi2 = select_features(X, y, method="chi2",        k=TOP_K_CHI2)
    X_mi,   sel_mi   = select_features(X, y, method="mutual_info", k=TOP_K_MI)
    print(f"\n  ► Using Chi2-selected features for model training.")

    # Train/test split on Chi2 features
    X_train, X_test, y_train, y_test = train_test_split(
        X_chi2, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n  Train size: {X_train.shape[0]:,}   Test size: {X_test.shape[0]:,}")

    # 5. Train & Evaluate
    results, fitted_models = train_and_evaluate(
        X_train, X_test, y_train, y_test,
        label_names=list(le.classes_)
    )

    # 6. Compare
    cmp_df, best_model_name = compare_models(results)

    # 7. Explainability
    best_model   = fitted_models[best_model_name]
    feat_names   = get_feature_names(tfidf, sel_chi2)
    top_features = explain_model(
        best_model, best_model_name, feat_names,
        label_names=list(le.classes_)
    )

    # 8. Visualisations
    print(f"\n{'='*60}")
    print("SECTION 8: VISUALISATIONS")
    print(f"{'='*60}")
    plot_sentiment_distribution(df)
    plot_review_length(df)
    plot_model_comparison(cmp_df)
    plot_confusion_matrices(results, label_names=list(le.classes_))
    plot_top_words(df)
    plot_feature_importance(top_features, best_model_name)

    # 9. Save
    save_artifacts(best_model_name, fitted_models, tfidf,
                   sel_chi2, le, cmp_df)

    elapsed = time.time() - t_start
    print(f"\n{'█'*60}")
    print(f"  PIPELINE COMPLETE  —  total time: {elapsed:.1f}s")
    print(f"  Best model : {best_model_name}")
    print(f"  Macro F1   : {cmp_df.iloc[0]['Macro F1']:.4f}")
    print(f"  Plots saved to   : '{PLOTS_DIR}/'")
    print(f"  Models saved to  : '{MODELS_DIR}/'")
    print("█" * 60 + "\n")

    return {
        "df"          : df,
        "cmp_df"      : cmp_df,
        "best_model"  : best_model_name,
        "results"     : results,
        "tfidf"       : tfidf,
        "selector"    : sel_chi2,
        "label_encoder": le,
    }


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("\nUsage: python sentiment_analysis.py <path_to_csv>")
        print("Example: python sentiment_analysis.py Amazon_Reviews.csv\n")
        sys.exit(1)

    csv_file = sys.argv[1]
    if not os.path.isfile(csv_file):
        print(f"\nERROR: File not found → '{csv_file}'\n")
        sys.exit(1)

    run_pipeline(csv_file)
