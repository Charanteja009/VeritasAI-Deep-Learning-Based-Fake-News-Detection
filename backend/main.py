import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from model_service import get_model_service

# Initialize FastAPI application
app = FastAPI(
    title="Fake News Detection API",
    description="Production-ready FastAPI backend for detecting Fake News using TF-IDF and Logistic Regression",
    version="1.0.0"
)

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request & Response Schemas
class NewsAnalysisRequest(BaseModel):
    title: Optional[str] = Field(default="", description="News headline or title")
    text: str = Field(..., description="Main news article content or statement")

class SampleItem(BaseModel):
    id: str
    category: str # "authentic" | "fake"
    title: str
    text: str
    description: str

# Curated benchmark samples from the dataset
SAMPLE_ARTICLES: List[SampleItem] = [
    SampleItem(
        id="sample-real-1",
        category="authentic",
        title="As U.S. budget fight looms, Republicans flip their fiscal script",
        text="WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a fiscal conservative on Sunday and pledged to demand deep federal spending curbs in 2018.",
        description="Factual Reuters political wire report with verified attribution."
    ),
    SampleItem(
        id="sample-real-2",
        category="authentic",
        title="U.S. military to accept transgender recruits on Monday: Pentagon",
        text="WASHINGTON (Reuters) - Transgender people will be allowed for the first time to enlist in the U.S. military starting on Monday as ordered by federal courts, the Pentagon said on Friday, after President Donald Trump's administration decided not to appeal rulings that blocked his transgender ban.",
        description="Official Pentagon announcement reported with neutral journalistic style."
    ),
    SampleItem(
        id="sample-fake-1",
        category="fake",
        title="SHOCKING: Leaked Internal Memo Exposes Worldwide Deep State Scheme!",
        text="You will not believe what anonymous whistleblower hackers just uncovered from subterranean servers! The corrupt mainstream media refuses to broadcast this emergency alert, but top secret leaked transcripts confirm corrupt politicians plan to rig the upcoming elections.",
        description="Sensational conspiracy article with exaggerated rhetoric and urgent clickbait phrasing."
    ),
    SampleItem(
        id="sample-fake-2",
        category="fake",
        title="Donald Trump Sends Out Embarrassing New Year's Eve Message; This is Disturbing",
        text="Donald Trump just couldn't wish all Americans a Happy New Year and leave it at that. Instead, he had to give a shout out to his enemies, haters and the very dishonest fake news media. The former reality TV star took to Twitter with a characteristically petty rant.",
        description="Highly opinionated blog editorial from the fake news corpus targeting political figures."
    )
]

# Root & Static Setup
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

@app.on_event("startup")
def startup_event():
    # Warm up model on server start
    print("[FastAPI] Pre-loading and warming up Fake News Model Service...")
    get_model_service()
    print("[FastAPI] Model ready to serve predictions.")

@app.get("/api/health")
def health_check():
    service = get_model_service()
    return {
        "status": "online",
        "model_loaded": service.is_loaded,
        "vocabulary_size": len(service.feature_names) if service.feature_names is not None else 0
    }

@app.get("/api/info")
def model_info():
    service = get_model_service()
    return {
        "model_name": "Logistic Regression Classifier",
        "vectorizer": "TF-IDF (10,000 features)",
        "classes": {
            "0": "Fake News",
            "1": "Authentic / Real News"
        },
        "model_file": service.model_path,
        "notebook_source": r"D:\2048\FakeNewsDetection.ipynb",
        "validation_metrics": {
            "accuracy": "99.0%",
            "precision": "99.0%",
            "recall": "99.0%",
            "f1_score": "0.99"
        }
    }

@app.get("/api/samples", response_model=List[SampleItem])
def get_sample_news():
    return SAMPLE_ARTICLES

@app.post("/api/predict")
def predict_news(payload: NewsAnalysisRequest):
    combined_len = len((payload.title or "").strip()) + len((payload.text or "").strip())
    if combined_len < 10:
        raise HTTPException(
            status_code=400,
            detail="Input is too short. Please provide at least 10 characters for meaningful NLP analysis."
        )

    try:
        service = get_model_service()
        result = service.predict(title=payload.title, text=payload.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files and fallback to index.html
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return {"message": "Frontend index.html not found"}
