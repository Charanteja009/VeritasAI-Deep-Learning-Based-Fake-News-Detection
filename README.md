# VeritasAI: Deep Learning-Based Fake News Detection System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow%20%2F%20Keras-2.15+-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Bi-LSTM Accuracy](https://img.shields.io/badge/Bi--LSTM%20Accuracy-99.90%25-success)](https://github.com/Charanteja009/VeritasAI-Deep-Learning-Based-Fake-News-Detection)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **VeritasAI** is an end-to-end Machine Learning and Deep Learning system designed to detect and classify fake, fabricated, and misleading news articles with high precision. The project compares a classical linear baseline against an advanced recurrent deep learning architecture, complete with an interactive, minimalist web evaluation dashboard.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Dataset Description](#-dataset-description)
3. [Text Preprocessing Pipeline](#-text-preprocessing-pipeline)
4. [Model Architectures Compared](#-model-architectures-compared)
   - [Baseline: TF-IDF + Logistic Regression](#1-baseline-tf-idf--logistic-regression)
   - [Deep Learning: Bidirectional LSTM (Bi-LSTM)](#2-deep-learning-bidirectional-lstm-bi-lstm)
5. [Comprehensive Performance Comparison](#-comprehensive-performance-comparison)
6. [Why Bidirectional LSTM Was Selected (Detailed Analysis)](#-why-bidirectional-lstm-was-selected-over-logistic-regression)
7. [Final Model Selection Decision](#-final-model-selection-decision)
8. [Web Interface & Architecture](#-web-interface--architecture)
9. [Installation & Quickstart Guide](#-installation--quickstart-guide)
10. [API Reference](#-api-reference)

---

## 🔍 Project Overview

The proliferation of online misinformation and sensationalized fake news presents significant challenges to public discourse, democratic processes, and media integrity. Manual fact-checking cannot keep pace with the velocity and volume of digital content.

This repository implements a rigorous empirical comparison between:
1. **Classical Machine Learning Baseline**: Term Frequency-Inverse Document Frequency (TF-IDF) feature extraction paired with an L2-regularized **Logistic Regression** classifier.
2. **Deep Sequence Model**: A multi-layer **Bidirectional Long Short-Term Memory (Bi-LSTM)** network with dense token embeddings, spatial dropout, and sequence masking.

The pipeline is wrapped in a high-performance **FastAPI** backend and an editorial **HTML5/CSS3/Vanilla JS** interface inspired by a minimalist *Paper & Stone* aesthetic.

---

## 📊 Dataset Description

The models were trained and benchmarked on the widely recognized **ISOT Fake News Dataset** (sourced from Kaggle / University of Victoria), comprising over 44,000 real-world news publications:

- **True / Authentic News (`label = 1`)**: 21,417 articles sourced directly from verified **Reuters** wire reporting covering domestic and international politics, diplomacy, and regulatory affairs.
- **Fake / Fabricated News (`label = 0`)**: 23,481 articles collected from unreliable, flagged, and debunked political blog feeds containing fabricated stories, unverified rumors, and clickbait narratives.

### Stratified Data Partitioning
To guarantee strict evaluation integrity and prevent data leakage:
- **Training Set (70%)**: 31,428 articles (strictly used for vocabulary adaptation, TF-IDF fitting, and model optimization)
- **Validation Set (15%)**: 6,735 articles (used for hyperparameter tuning and early stopping regularization)
- **Test Set (15%)**: 6,735 articles (held-out exclusively for final benchmark evaluation)

---

## ⚙️ Text Preprocessing Pipeline

Both models consume the combined article headline (`title`) and body (`text`) to maximize contextual signal:

$$\text{full\_text} = \text{title} + \text{" "} + \text{text}$$

The raw text undergoes standardized normalization:
1. **Case Normalization**: Conversion of all characters to lowercase.
2. **Regex Tag Stripping**: Removal of HTML tags (`<[^>]+>`) and markup noise.
3. **Hyperlink Removal**: Elimination of URLs (`https?://\S+|www\.\S+`).
4. **Punctuation Translation**: Removal of punctuation marks (`string.punctuation`) while preserving word boundaries.
5. **Whitespace Regularization**: Collapsing multiple whitespace gaps, carriage returns, and tabs into a single clean space.

---

## 🧠 Model Architectures Compared

### 1. Baseline: TF-IDF + Logistic Regression
- **Feature Extraction**: Scikit-Learn `TfidfVectorizer` with `max_features=10,000`. Computes term-frequency inverse document frequency weights across unigram tokens.
- **Classification Engine**: `LogisticRegression` with L2-regularization penalty, `C=1.0`, `solver='lbfgs'`, and `max_iter=1000`.
- **Inference Mechanism**: Linear combination of TF-IDF feature weights followed by a logistic sigmoid function:
  $$P(y=1|\mathbf{x}) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **Key Characteristics**: Extremely fast CPU training and sub-millisecond inference, but inherently ignores word ordering and temporal relationships (Bag-of-Words assumption).

---

### 2. Deep Learning: Bidirectional LSTM (Bi-LSTM)
The champion deep learning architecture is constructed using **TensorFlow/Keras**:

```text
 ┌─────────────────────────────────────────────────────────────┐
 │  Input Layer (tf.string: Raw normalized text sequence)      │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │  TextVectorization (max_tokens=10,000, sequence_length=300) │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │  Embedding Layer (dim=128, mask_zero=True, trainable=True)  │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │  Bidirectional LSTM Layer (64 forward + 64 backward units)  │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │  Spatial Dropout Layer (rate = 0.5)                         │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │  Dense Hidden Layer (32 units, ReLU activation)             │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │  Dropout Layer (rate = 0.5)                                 │
 └──────────────────────────────┬──────────────────────────────┘
                                │
 ┌──────────────────────────────▼──────────────────────────────┐
 │  Output Classification Layer (1 unit, Sigmoid activation)   │
 └─────────────────────────────────────────────────────────────┘
```

- **Vectorization**: Truncates or pads articles to a fixed sequence length of 300 tokens.
- **Dense Embeddings**: Maps each of the 10,000 vocabulary tokens into a continuous 128-dimensional dense vector space. Masking (`mask_zero=True`) ignores zero-padded time steps.
- **Bidirectional Processing**: Employs two independent recurrent states—one processing the document forward from token $1 \to T$, and the other processing backward from $T \to 1$.
- **Regularization & Optimization**:
  - Dropout layers (0.5 rate) to prevent co-adaptation of neurons and eliminate overfitting.
  - Optimized with **Adam** ($\text{learning rate} = 10^{-3}$) under **Binary Cross-Entropy** loss.
  - **EarlyStopping** callback monitoring validation loss (`patience=2`, `restore_best_weights=True`).
  - **ModelCheckpoint** persisting the best weight tensor.

---

## 📈 Comprehensive Performance Comparison

Both models were evaluated on the identical stratified test partition (6,735 unseen articles). Below is the comparative benchmark:

| Metric | Baseline: TF-IDF + Logistic Regression | Champion: Bidirectional LSTM | Delta (&Delta;) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 99.00% | **99.90%** | **+0.90%** |
| **Precision** | 99.00% | **99.97%** | **+0.97%** |
| **Recall** | 99.00% | **99.81%** | **+0.81%** |
| **F1-Score** | 0.9900 | **0.9989** | **+0.0089** |
| **Area Under ROC Curve (AUC)** | 0.9998 | **0.9999** | **+0.0001** |
| **Total Test Misclassifications** | ~67 articles | **Only 7 articles** | **90% Error Reduction** |
| **Feature Representation** | Sparse Bag-of-Words (Sparse 10k) | Dense Semantic Embeddings (128-dim) | Continuous Semantic Space |
| **Word Order Sensitivity** | ❌ None (Position-invariant) | ✅ Full (Bidirectional Sequence Context) | Syntactic & Grammatical Flow |
| **Handling of Negations** | ❌ Prone to bag-of-words ambiguity | ✅ Preserves grammatical dependency | Robust to inverted meanings |

### Classification Reports

```text
=== Baseline: Logistic Regression (Test Set) ===
              precision    recall  f1-score   support
    Fake (0)       0.99      0.99      0.99      3522
    True (1)       0.99      0.99      0.99      3213
    accuracy                           0.99      6735
   macro avg       0.99      0.99      0.99      6735
weighted avg       0.99      0.99      0.99      6735

=== Champion: Bidirectional LSTM (Test Set) ===
              precision    recall  f1-score   support
    Fake (0)       1.00      1.00      1.00      3522
    True (1)       1.00      1.00      1.00      3213
    accuracy                           1.00      6735
   macro avg       1.00      1.00      1.00      6735
weighted avg       1.00      1.00      1.00      6735
```

---

## 🏆 Why Bidirectional LSTM Was Selected Over Logistic Regression

While the baseline Logistic Regression model achieved a strong 99.0% accuracy, **the Bidirectional LSTM was decisively selected as the champion model** for production deployment and deeper natural language analysis for four foundational scientific reasons:

### 1. Bidirectional Temporal & Sequential Context Awareness
- **Limitation of Logistic Regression**: TF-IDF relies on the **Bag-of-Words (BoW)** assumption. It treats text as an unordered collection of token frequencies. The phrases *"The senator did not vote for the illegal bribery bill"* and *"The senator did vote for the illegal bribery bill, not..."* yield nearly identical sparse vector representations despite having diametrically opposite semantic meanings.
- **Bi-LSTM Advantage**: The Bidirectional LSTM processes tokens sequentially through time $t$. At any token $x_t$, the forward hidden state $\vec{h}_t$ encodes the narrative history leading up to $x_t$, while the backward hidden state $\overleftarrow{h}_t$ encodes the narrative following $x_t$. The concatenated representation:
  $$h_t = \left[ \vec{h}_t \,\|\, \overleftarrow{h}_t \right]$$
  enables the network to decipher the full contextual horizon, detecting subtleties, sarcasm, negations, and complex rhetorical structures characteristic of sophisticated misinformation.

### 2. Continuous Dense Embeddings vs. Sparse Orthogonal Vectors
- **Limitation of Logistic Regression**: TF-IDF vectors are discrete and orthogonal. If an article uses the synonym *"fabricated"* instead of *"fake"*, or *"unsubstantiated"* instead of *"false"*, TF-IDF treats them as completely independent features with zero mathematical overlap.
- **Bi-LSTM Advantage**: The 128-dimensional continuous embedding layer maps tokens into a dense manifold where semantically related words are clustered together via learned geometric proximity. Even when facing novel vocabulary combinations or paraphrased clickbait narratives, the Bi-LSTM generalizes with superior robustness.

### 3. A 90% Reduction in False Positives & Misclassifications
- In real-world content moderation, **false accusations of disinformation carry grave consequences** for journalistic credibility.
- On the held-out test split of 6,735 articles, the TF-IDF Logistic Regression model produced approximately 67 classification errors.
- The Bidirectional LSTM reduced total errors down to **just 7 articles**, achieving an exceptional **99.97% Precision**. This represents an **order-of-magnitude (90%) reduction in error rate**, making the deep learning model far safer for deployment.

### 4. Robust Long-Range Dependency Retention
- Deceptive articles frequently bury clickbait or fabricated claims inside real-looking introductions, or append sensational calls-to-action at the end of an otherwise standard paragraph.
- Traditional recurrent networks (RNNs) suffer from vanishing gradients over sequences longer than 10–20 words.
- The LSTM architecture solves this via **input, forget, and output gates** ($i_t, f_t, o_t$) and an internal memory cell state ($c_t$):
  $$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$
  $$c_t = f_t * c_{t-1} + i_t * \tilde{c}_t$$
  This mechanism allows the network to carry salient warning cues across the full 300-word sequence window without attenuation.

---

## 🎯 Final Model Selection Decision

```text
================================================================================
                          FINAL MODEL SELECTION SYNTHESIS
================================================================================

 [BASELINE CANDIDATE]                           [SELECTED CHAMPION]
  TF-IDF + Logistic Regression                   Bidirectional LSTM Deep Network
  - Accuracy:  99.00%                            - Accuracy:  99.90%  (+0.90%)
  - Precision: 99.00%                            - Precision: 99.97%  (+0.97%)
  - Recall:    99.00%                            - Recall:    99.81%  (+0.81%)
  - F1-Score:  0.9900                            - F1-Score:  0.9989  (+0.0089)
  - Error:     ~67 misclassifications            - Error:     Only 7 misclassifications

  PRIMARY LIMITATION:                            PRIMARY ADVANTAGE:
  Bag-of-Words loses syntax, grammar,            Dual-direction recurrent states capture
  negations, and temporal word order.            word order, semantic nuances & context.

================================================================================
 CONCLUSION: The Bidirectional LSTM is officially crowned as the production-grade 
 architecture for VeritasAI due to its near-perfect discriminative capability, 
 semantic depth, and 90% reduction in real-world classification errors.
================================================================================
```

> **Note on Deployment**: For rapid local demonstration and instant CPU testing without requiring heavy GPU hardware or TensorFlow installations, the repository also includes the lightweight, pre-exported `best_logistic_regression_pipeline.pkl` which runs in under **4 milliseconds** on standard laptop processors. Both pipelines share identical preprocessing logic.

---

## 💻 Web Interface & Architecture

The accompanying web dashboard provides an interactive editorial experience designed with a **Paper, Stone, Coral, and Black** aesthetic:

- **Frontend**: Clean semantic HTML5, modern CSS3 (with responsive layout, smooth state transitions, and custom typography using *Newsreader*, *Plus Jakarta Sans*, and *Space Mono*), and asynchronous Vanilla JavaScript (`app.js`).
- **Backend**: Built with **FastAPI** (`main.py`) offering sub-10ms latency, CORS middleware, strict Pydantic payload validation, and automatic OpenAPI Swagger documentation.
- **Explainable AI (XAI)**: Visualizes exact token attribution scores—highlighting words that pushed the prediction towards Authentic (Black pills) vs. Fake (Coral pills).
- **Interactive Ratio Split Bar**: Displays real-time dual-class percentage distributions.
- **One-Click Benchmark Triggers**: Integrated quick-test sample dispatches (Reuters policy reports vs. viral conspiracy hoaxes).

---

## 🚀 Installation & Quickstart Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/Charanteja009/VeritasAI-Deep-Learning-Based-Fake-News-Detection.git
cd VeritasAI-Deep-Learning-Based-Fake-News-Detection
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Dependencies
```bash
pip install -r requirements.txt
```

*(Optional: To re-train or inspect the deep learning notebook locally, install TensorFlow: `pip install tensorflow matplotlib seaborn pandas`)*

### 4. Run the Web Application
```bash
# Option A: Python command
python run.py

# Option B: Windows 1-Click Batch file
run.bat
```

Once running, access the services in your browser:
- **Web Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Redoc API Documentation:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 📡 API Reference

### `POST /api/predict`
Evaluates the authenticity of an article or headline.

**Request Body:**
```json
{
  "title": "U.S. Federal Reserve holds benchmark interest rates steady amid inflation data",
  "text": "WASHINGTON (Reuters) - The Federal Reserve kept interest rates unchanged on Wednesday, pointing to sustained economic expansion and moderate job growth while inflation continues to approach the central bank target rate."
}
```

**Response Payload:**
```json
{
  "prediction": 1,
  "label": "Authentic News",
  "is_fake": false,
  "confidence": 99.18,
  "probabilities": {
    "fake": 0.82,
    "real": 99.18
  },
  "metrics": {
    "word_count": 37,
    "char_count": 251,
    "latency_ms": 3.8
  },
  "explainability": {
    "fake_signals": [
      { "word": "continues", "weight": 0.0821, "direction": "fake" }
    ],
    "real_signals": [
      { "word": "reuters", "weight": 2.0352, "direction": "real" },
      { "word": "washington", "weight": 0.6885, "direction": "real" },
      { "word": "wednesday", "weight": 0.4215, "direction": "real" }
    ]
  },
  "cleaned_preview": "us federal reserve holds benchmark interest rates steady amid inflation data washington reuters..."
}
```

### `GET /api/samples`
Returns benchmark news articles for one-click testing.

### `GET /api/health`
Returns the status of the server and model vectorizer features.

---

## 📂 Repository File Structure

```
VeritasAI-Deep-Learning-Based-Fake-News-Detection/
│
├── FakeNewsDetection.ipynb            # Original Jupyter notebook with EDA, Bi-LSTM training, and evaluation
├── best_logistic_regression_pipeline.pkl # Pre-trained serialized inference pipeline (TF-IDF + LogReg)
│
├── backend/
│   ├── main.py                       # FastAPI application, static routing, and endpoints
│   └── model_service.py              # Model loader, NLP text cleaning, inference, and XAI attribution
│
├── frontend/
│   ├── index.html                    # Minimalist Paper & Stone dashboard UI
│   ├── style.css                     # Custom design system (Paper, Stone, Coral, Black)
│   └── app.js                        # Asynchronous fetch client, dynamic split ratio bar, and telemetry
│
├── run.py                            # Python application launcher with automatic browser opening
├── run.bat                           # 1-click Windows batch launcher
├── requirements.txt                  # Python dependencies for the web engine
├── .gitignore                        # Git exclusion rules
└── README.md                         # Detailed project documentation and model comparison
```

---

## 📜 License
This project is licensed under the [MIT License](LICENSE).

## 👤 Author
- **Charan Teja** ([@Charanteja009](https://github.com/Charanteja009))
