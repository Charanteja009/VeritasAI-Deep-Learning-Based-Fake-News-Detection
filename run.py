import os
import sys
import webbrowser
from pathlib import Path

# Add backend directory to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "backend"))

if __name__ == "__main__":
    import uvicorn

    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"

    print("=" * 60)
    print(" 📰 VeritasAI - Fake News Detection Web Application")
    print("=" * 60)
    print(f"Server starting on: {url}")
    print(f"FastAPI Docs:       {url}/docs")
    print(f"Model Pipeline:     D:\\2048\\best_logistic_regression_pipeline.pkl")
    print("=" * 60)
    print("Press Ctrl+C to stop the server.")

    # Automatically open browser in desktop environment
    try:
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("main:app", host=host, port=port, app_dir=str(BASE_DIR / "backend"), reload=False)
