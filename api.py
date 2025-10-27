from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from llama_cpp import Llama
from transformers import pipeline
import requests
from xml.etree import ElementTree as ET
import re
from typing import List, Dict
import os
import json
import asyncio

app = FastAPI(title="News & Fact API")

LOG_DIR = "news_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def sanitize_filename(s: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '_', s)

# -------------------------
# News API (unchanged)
# -------------------------
class NewsAPI:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        })

    def search_news(self, query: str, max_results: int = 10) -> List[Dict]:
        url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"News API error: {e}")
            return []

        try:
            root = ET.fromstring(r.content)
        except ET.ParseError as e:
            print(f"XML parse error: {e}")
            return []

        articles = []
        for item in root.findall(".//item")[:max_results]:
            source = item.find("source")
            desc = item.find("description")
            link = item.find("link")  # Get the article URL
            
            if desc is None or not desc.text:
                continue
            
            content = re.sub(r"<[^>]+>", " ", desc.text)
            content = re.sub(r"&[a-z]+;", " ", content)
            content = re.sub(r"\s+", " ", content).strip()
            content = re.sub(r"^(.*?)(?:-|\|).{0,100}", "", content).strip()
            
            if len(content) < 50:
                continue
            
            articles.append({
                "source": (source.text if source is not None else "Unknown").strip(),
                "content": content,
                "url": (link.text if link is not None else "")  # Add URL
            })

        filename = sanitize_filename(f"{query}_raw.json")
        with open(os.path.join(LOG_DIR, filename), "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        return articles

# -------------------------
# News QA with Streaming Support
# -------------------------
class NewsQueryAnswerer:
    def __init__(self, model_path: str, gpu_layers: int = 0):
        if not model_path:
            raise ValueError("Local model path must be provided.")
        
        print("Loading Mistral model with llama-cpp-python...")
        
        if gpu_layers > 0:
            print(f"🎮 GPU acceleration enabled - offloading {gpu_layers} layers to Intel GPU")
        
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=gpu_layers,
            n_ctx=1024,
            n_threads=os.cpu_count() or 4,
            n_batch=512,
            verbose=False
        )
        
        print("✓ Mistral model loaded successfully")
        self.news_api = NewsAPI()

    def prepare_context(self, articles: List[Dict], query: str) -> str:
        lines = []
        for idx, a in enumerate(articles):
            content = a["content"]
            sentences = [s.strip() + "." if not s.strip().endswith(('.', '!', '?')) else s.strip()
                        for s in re.split(r'(?<=[.!?])\s+', content)
                        if len(s.strip()) > 30]
            snippet = " ".join(sentences[:2])
            lines.append(f"Article {idx+1}: {snippet}")

        context_lines = []
        word_count = 0
        for line in lines:
            line_words = line.split()
            if word_count + len(line_words) > 400:
                line_words = line_words[:400 - word_count]
            context_lines.append(" ".join(line_words))
            word_count += len(line_words)
            if word_count >= 400:
                break

        context = "\n".join(context_lines)

        filename = sanitize_filename(f"{query}_context.txt")
        with open(os.path.join(LOG_DIR, filename), "w", encoding="utf-8") as f:
            f.write(context)

        return context

    def generate_stream(self, query: str, context: str):
        """Generate streaming response for SSE"""
        prompt = f"""[INST] You are a factual news assistant.
Read the following latest news and answer the query directly and succinctly.
Confirm whether the latest news agree with the query and provide summarized context to explain your answer.

Context:
{context}

Query: {query} [/INST]

Answer:"""
        
        try:
            response = self.model(
                prompt,
                max_tokens=200,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["[INST]", "</s>"],
                stream=True  # Enable streaming
            )
            
            for chunk in response:
                token = chunk['choices'][0]['text']
                if token:
                    yield f"data: {json.dumps({'token': token})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            
        except Exception as e:
            print(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

# Initialize models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MISTRAL_PATH = os.path.join(BASE_DIR, 'models', 'Mistral-7B-Instruct-v0.2-GGUF', 'mistral-7b-instruct-v0.2.Q4_K_M.gguf')
BART_PATH = os.path.join(BASE_DIR, 'models', 'bart-large-mnli')

print("Initializing News QA system with Mistral GGUF (Intel GPU)...")
qa = NewsQueryAnswerer(MISTRAL_PATH, gpu_layers=32)
print("News QA system ready!")

print("Loading BART zero-shot classifier for worthiness checking...")
classifier = pipeline("zero-shot-classification", model=BART_PATH)
print("✓ BART classifier loaded successfully")

# Themes for worthiness checking
THEMES = {
    "THEME 1 (NEWS/JOURNALISM)": {"worthy": "news announcement", "unworthy": "personal feeling"},
    "THEME 2 (OBJECTIVITY)": {"worthy": "logical", "unworthy": "absurd"},
    "THEME 3 (INSTITUTIONAL/SCOPE)": {"worthy": "institutional", "unworthy": "personal"},
    "THEME 4 (ACTION vs COGNITION)": {"worthy": "happening", "unworthy": "thinking"},
}

def is_worthy(claim: str) -> Dict:
    try:
        normalized = claim.strip()
        theme_results = []
        votes_worthy = 0
        
        for theme_name, labels in THEMES.items():
            candidate_labels = [labels["worthy"], labels["unworthy"]]
            result = classifier(normalized, candidate_labels=candidate_labels)
            
            top_label = result["labels"][0]
            top_score = result["scores"][0]
            is_theme_worthy = (top_label == labels["worthy"])
            
            if is_theme_worthy:
                votes_worthy += 1
            
            theme_results.append({
                "theme": theme_name,
                "predicted_label": top_label,
                "score": top_score,
                "worthy": is_theme_worthy
            })
        
        final_decision = (votes_worthy >= 3)
        
        if final_decision:
            reason = f"✅ WORTHY: {votes_worthy}/4 themes voted worthy (majority reached)"
        else:
            reason = f"❌ UNWORTHY: Only {votes_worthy}/4 themes voted worthy (need >= 3)"
        
        return {
            "is_worthy": final_decision,
            "reason": reason,
            "votes_worthy": votes_worthy,
            "votes_needed": 3,
            "theme_results": theme_results,
            "method": "COMBO-MAJORITY (F1: 93.0%)"
        }
        
    except Exception as e:
        print(f"Worthiness check error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "is_worthy": True,
            "reason": f"⚠️ Error during classification: {str(e)}. Defaulting to worthy.",
            "votes_worthy": 0,
            "votes_needed": 3,
            "theme_results": [],
            "method": "COMBO-MAJORITY (F1: 93.0%)"
        }

# API Models
class ClaimRequest(BaseModel):
    claim: str

class QueryRequest(BaseModel):
    query: str
    max_articles: int = 10

# API Endpoints
@app.get("/")
def root():
    return {
        "message": "News & Fact API with Streaming Support",
        "endpoints": {
            "/is_worthy": "POST - Check if claim is worthy",
            "/summarize": "POST - Get news summary (streaming SSE)"
        }
    }

@app.post("/is_worthy")
def api_is_worthy(req: ClaimRequest):
    result = is_worthy(req.claim)
    return {
        "claim": req.claim,
        **result
    }

@app.post("/summarize")
async def api_summarize(req: QueryRequest):
    """
    Streaming endpoint using Server-Sent Events (SSE).
    Returns tokens as they are generated.
    """
    try:
        # Search articles first
        articles = qa.news_api.search_news(req.query, max_results=req.max_articles)
        
        if not articles:
            async def no_news_stream():
                yield f"data: {json.dumps({'error': 'No news found'})}\n\n"
            
            return StreamingResponse(
                no_news_stream(),
                media_type="text/event-stream"
            )
        
        # Prepare context
        context = qa.prepare_context(articles, req.query)
        
        # Group articles by source domain and keep URLs
        sources_dict = {}
        for article in articles:
            source = article["source"]
            url = article.get("url", "")
            if source not in sources_dict:
                sources_dict[source] = url
        
        # Send metadata first
        async def stream_with_metadata():
            # Send metadata with sources and their URLs
            metadata = {
                "type": "metadata",
                "sources": sources_dict,  # Now sends {source: url} mapping
                "num_articles": len(articles)
            }
            yield f"data: {json.dumps(metadata)}\n\n"
            
            # Stream tokens
            for chunk in qa.generate_stream(req.query, context):
                yield chunk
                await asyncio.sleep(0)  # Allow other tasks to run
        
        return StreamingResponse(
            stream_with_metadata(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
        
    except Exception as e:
        print(f"Error in summarize endpoint: {e}")
        import traceback
        traceback.print_exc()
        
        async def error_stream():
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        
        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream"
        )
    
