"""LLM wrapper with on-disk cache (saves API quota during evaluation reruns)."""
import hashlib, json, os, re, requests
from config import LLM_PROVIDER, GEMINI_MODEL, OLLAMA_MODEL

CACHE_PATH = "cache/llm_cache.json"
_cache = json.load(open(CACHE_PATH)) if os.path.exists(CACHE_PATH) else {}

def _save():
    os.makedirs("cache", exist_ok=True)
    json.dump(_cache, open(CACHE_PATH, "w"))

def generate(prompt: str, system: str = "") -> str:
    key = hashlib.sha256(f"{LLM_PROVIDER}|{GEMINI_MODEL}|{system}|{prompt}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    if LLM_PROVIDER == "ollama":
        r = requests.post("http://localhost:11434/api/generate",
                          json={"model": OLLAMA_MODEL, "prompt": prompt, "system": system,
                                "stream": False, "options": {"temperature": 0.1}}, timeout=300)
        text = r.json()["response"]
    else:
        from google import genai
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        r = client.models.generate_content(model=GEMINI_MODEL, contents=prompt,
                                           config={"system_instruction": system, "temperature": 0.1})
        text = r.text
    _cache[key] = text
    _save()
    return text

def generate_json(prompt: str, system: str = ""):
    raw = generate(prompt + "\n\nReturn ONLY valid JSON, no markdown fences.", system)
    m = re.search(r"(\{.*\}|\[.*\])", raw, re.S)
    try:
        return json.loads(m.group(1)) if m else None
    except json.JSONDecodeError:
        return None

def generate_plain_baseline(question: str) -> str:
    """Baseline: LLM with NO retrieval (for the 'plain LLM' comparison)."""
    return generate(f"Answer this research question concisely:\n{question}")
