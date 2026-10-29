import os
import sys
import re
import string
import time
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, List

# Suppress scikit-learn version mismatch warnings during unpickling
warnings.filterwarnings("ignore", category=UserWarning)

# Preprocessing function matching the original Colab notebook training exactly
def clean_text(text: str) -> str:
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove URLs
    text = re.sub(r'https?:\/\/\S+|www\.\s+', '', text)
    # Remove punctuation marks
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    # Collapse multiple whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# The original pickle serialized clean_text under __main__; ensure __main__ contains it
main_mod = sys.modules.get('__main__')
if main_mod is not None:
    setattr(main_mod, 'clean_text', clean_text)


class FakeNewsModelService:
    def __init__(self, model_path: str = None):
        self.model_path = model_path or self._resolve_default_model_path()
        self.vectorizer = None
        self.model = None
        self.feature_names = None
        self.is_loaded = False
        self.load_model()

    def _resolve_default_model_path(self) -> str:
        candidates = [
            Path(__file__).resolve().parent.parent / "best_logistic_regression_pipeline.pkl",
            Path.cwd() / "best_logistic_regression_pipeline.pkl",
            r"D:\2048\best_logistic_regression_pipeline.pkl",
            Path(__file__).resolve().parent.parent / "models" / "best_logistic_regression_pipeline.pkl"
        ]
        for candidate in candidates:
            if os.path.exists(str(candidate)):
                return str(candidate)
        return r"D:\2048\best_logistic_regression_pipeline.pkl"

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        # Inject clean_text into __main__ right before unpickling
        import __main__
        __main__.clean_text = clean_text

        with open(self.model_path, "rb") as f:
            pipeline = pickle.load(f)

        self.vectorizer = pipeline["vectorizer"]
        self.model = pipeline["model"]
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_loaded = True
        print(f"[ModelService] Pipeline loaded successfully from {self.model_path}")
        print(f"[ModelService] Vocabulary size: {len(self.feature_names)} features")

    def predict(self, title: str = "", text: str = "") -> Dict[str, Any]:
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded.")

        start_time = time.perf_counter()

        title_clean = (title or "").strip()
        text_clean = (text or "").strip()

        # Combine title and text as done in notebook preprocessing
        full_text = f"{title_clean} {text_clean}".strip()
        if not full_text:
            raise ValueError("Input text cannot be empty.")

        cleaned = clean_text(full_text)
        words = cleaned.split()
        word_count = len(words)
        char_count = len(cleaned)

        # TF-IDF vectorization and Logistic Regression prediction
        transformed = self.vectorizer.transform([cleaned])
        pred_code = int(self.model.predict(transformed)[0]) # 1: True / Real, 0: Fake
        proba = self.model.predict_proba(transformed)[0]

        fake_prob = float(proba[0])
        real_prob = float(proba[1])

        is_fake = (pred_code == 0)
        verdict_label = "Fake News" if is_fake else "Authentic News"
        confidence = fake_prob if is_fake else real_prob

        # Explainability: Extract feature impact for active tokens
        nonzero_indices = transformed.nonzero()[1]
        tfidf_values = transformed.data
        coefficients = self.model.coef_[0][nonzero_indices]
        contributions = tfidf_values * coefficients

        fake_indicators: List[Dict[str, Any]] = []
        real_indicators: List[Dict[str, Any]] = []

        if len(nonzero_indices) > 0:
            import numpy as np
            sort_indices = np.argsort(contributions)

            # Negative contribution pulls toward Fake (Class 0)
            for idx in sort_indices:
                if contributions[idx] < 0:
                    word = str(self.feature_names[nonzero_indices[idx]])
                    fake_indicators.append({
                        "word": word,
                        "weight": round(float(abs(contributions[idx])), 4),
                        "direction": "fake"
                    })
                if len(fake_indicators) >= 6:
                    break

            # Positive contribution pulls toward Real (Class 1)
            for idx in reversed(sort_indices):
                if contributions[idx] > 0:
                    word = str(self.feature_names[nonzero_indices[idx]])
                    real_indicators.append({
                        "word": word,
                        "weight": round(float(contributions[idx]), 4),
                        "direction": "real"
                    })
                if len(real_indicators) >= 6:
                    break

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "prediction": pred_code,
            "label": verdict_label,
            "is_fake": is_fake,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "fake": round(fake_prob * 100, 2),
                "real": round(real_prob * 100, 2)
            },
            "metrics": {
                "word_count": word_count,
                "char_count": char_count,
                "latency_ms": latency_ms
            },
            "explainability": {
                "fake_signals": fake_indicators,
                "real_signals": real_indicators
            },
            "cleaned_preview": cleaned[:300] + ("..." if len(cleaned) > 300 else "")
        }

# Global singleton
service = None

def get_model_service() -> FakeNewsModelService:
    global service
    if service is None:
        service = FakeNewsModelService()
    return service
