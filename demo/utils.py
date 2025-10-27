from transformers import pipeline

model_a = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def is_worthy(claim: str) -> bool:
    personal_result = model_a(claim, candidate_labels=["objective news", "personal opinion"])
    if personal_result['labels'][0] == "personal opinion":
        return False
    
    joke_result = model_a(claim, candidate_labels=["credible claim", "absurd statement"])
    if joke_result['labels'][0] == "absurd statement":
        return False
    
    sig_result = model_a(claim, candidate_labels=["newsworthy event", "mundane occurrence"])
    if sig_result['labels'][0] != "newsworthy event":
        return False
    
    return True

# news_api_module.py
import requests

API_KEY = "565a1816deae427f9842b4b1f5cbdd95"

def fetch_latest_news(query, page_size=5):
    url = f"https://newsapi.org/v2/top-headlines?language=en&q={query}&pageSize={page_size}&apiKey={API_KEY}"
    resp = requests.get(url)
    data = resp.json()
    articles = data.get("articles", [])
    # Compact representation: title + description + source
    compact_news = [
        {
            "title": art.get("title"),
            "description": art.get("description"),
            "source": art.get("source", {}).get("name"),
            "url": art.get("url")
        } for art in articles
    ]
    return compact_news
