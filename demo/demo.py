import requests
from xml.etree import ElementTree as ET
import re
from typing import List, Dict
import os
import json
from llama_cpp import Llama
import sys

# -------------------------
# Config
# -------------------------
LOG_DIR = "news_logs"
os.makedirs(LOG_DIR, exist_ok=True)

def sanitize_filename(s: str) -> str:
    """Replace invalid Windows filename characters."""
    return re.sub(r'[<>:"/\\|?*]', '_', s)

# -------------------------
# News API
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
            if desc is None or not desc.text:
                continue
            # Clean HTML & entities
            content = re.sub(r"<[^>]+>", " ", desc.text)
            content = re.sub(r"&[a-z]+;", " ", content)
            content = re.sub(r"\s+", " ", content).strip()
            content = re.sub(r"^(.*?)(?:-|\|).{0,100}", "", content).strip()
            if len(content) < 50:
                continue
            articles.append({
                "source": (source.text if source is not None else "Unknown").strip(),
                "content": content
            })

        # Save raw articles
        filename = sanitize_filename(f"{query}_raw.json")
        with open(os.path.join(LOG_DIR, filename), "w", encoding="utf-8") as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

        return articles

# -------------------------
# News QA
# -------------------------
class NewsQueryAnswerer:
    def __init__(self, model_path: str, gpu_layers: int = 0):
        if not model_path:
            raise ValueError("Local model path must be provided.")
        
        print("Loading model with llama-cpp-python (this may take a few minutes)...")
        
        if gpu_layers > 0:
            print(f"🎮 GPU acceleration enabled - offloading {gpu_layers} layers to Intel GPU")
        
        # Load GGUF model with llama-cpp-python - INTEL GPU SUPPORT
        self.model = Llama(
            model_path=model_path,
            n_gpu_layers=gpu_layers,  # Offload layers to GPU (0=CPU, 32=full GPU)
            n_ctx=1024,  # Context window
            n_threads=os.cpu_count() or 4,  # CPU threads for non-GPU work
            n_batch=512,  # Batch size
            verbose=True  # Show loading details
        )
        
        print("✓ Model loaded successfully")
        self.news_api = NewsAPI()

    def answer_query(self, query: str, max_articles: int = 10) -> Dict:
        print(f"\nSearching news for: {query}")
        articles = self.news_api.search_news(query, max_results=max_articles)
        if not articles:
            return {"query": query, "answer": "No news found.", "sources": [], "num_articles": 0, "articles": []}

        print(f"Found {len(articles)} articles. Generating answer...")
        context = self._prepare_context(articles, query)
        answer = self._generate_answer(query, context)
        sources = list({a["source"] for a in articles})
        return {"query": query, "answer": answer, "sources": sources, "num_articles": len(articles), "articles": articles}

    def _prepare_context(self, articles: List[Dict], query: str) -> str:
        lines = []
        for idx, a in enumerate(articles):
            content = a["content"]
            sentences = [s.strip() + "." if not s.strip().endswith(('.', '!', '?')) else s.strip()
                        for s in re.split(r'(?<=[.!?])\s+', content)
                        if len(s.strip()) > 30]
            snippet = " ".join(sentences[:2])
            lines.append(f"Article {idx+1}: {snippet}")

        # Limit total words to 400
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

        # Save context
        filename = sanitize_filename(f"{query}_context.txt")
        with open(os.path.join(LOG_DIR, filename), "w", encoding="utf-8") as f:
            f.write(context)

        return context

    def _generate_answer(self, query: str, context: str) -> str:
        # Mistral-Instruct format
        prompt = f"""[INST] You are a factual news assistant.
Read the following latest news and answer the query directly and succinctly.
Confirm whether the latest news agree with the query and provide summarized context to explain your answer.

Context:
{context}

Query: {query} [/INST]

Answer:"""
        
        try:
            print("\n🤖 AI Response: ", end="", flush=True)
            
            # Collect tokens for streaming display
            full_text = ""
            
            # Generate with streaming enabled
            response = self.model(
                prompt,
                max_tokens=200,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["[INST]", "</s>"],
                stream=True  # Enable streaming!
            )
            
            for chunk in response:
                token = chunk['choices'][0]['text']
                print(token, end="", flush=True)
                full_text += token
            
            print()  # New line after generation
            
            return self._clean_response(full_text)
        except Exception as e:
            print(f"\nGeneration error: {e}")
            import traceback
            traceback.print_exc()
            return "Error generating answer."
  
    def _clean_response(self, text: str) -> str:
        text = re.sub(r'^(Answer:|Context:|Question:)\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        if text and not text[-1] in '.!?':
            text += '.'
        return text

    def display_result(self, result: Dict):
        print(f"\n{'='*60}")
        print(f"Query: {result['query']}")
        print(f"{'='*60}")
        print(f"\n✓ Final Answer: {result['answer']}")
        print(f"\n📰 Sources ({result['num_articles']} articles): {', '.join(result['sources'][:5])}")
        if len(result['sources']) > 5:
            print(f"  ... and {len(result['sources']) - 5} more")
        print(f"\n{'-'*60}")

# -------------------------
# Demo
# -------------------------
def demo():
    # INTEL GPU CONFIGURATION
    # Start with gpu_layers=32 for Intel GPU acceleration via SYCL
    # Reduce to 16 or 24 if you get memory errors
    qa = NewsQueryAnswerer(
        r'modern_issues_demo\models\Mistral-7B-Instruct-v0.2-GGUF\mistral-7b-instruct-v0.2.Q4_K_M.gguf',
        gpu_layers=32  # Set to 32 for Intel GPU (SYCL backend)
    )
    
    queries = [
        "Was Trump arrested yesterday?",
        "Climate change has hit all time severity",
        "Elon Musk is one of the richest man on earth",
    ]
    
    for q in queries:
        try:
            result = qa.answer_query(q, max_articles=5)
            qa.display_result(result)
        except Exception as e:
            print(f"Error processing query '{q}': {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    demo()