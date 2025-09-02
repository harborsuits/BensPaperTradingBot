from __future__ import annotations
from typing import List, Dict, Any, Tuple
import time
import re
import httpx
import os
import yaml
from urllib.parse import urlparse, urlunparse

try:
    import feedparser
except Exception:
    feedparser = None

try:
    from trafilatura import extract as trafi_extract
except Exception:
    trafi_extract = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # optional
    _VADER = SentimentIntensityAnalyzer()
except Exception:
    _VADER = None

try:
    from rapidfuzz import fuzz
except Exception:
    # Simple fallback if rapidfuzz is not available
    class FallbackFuzz:
        def token_set_ratio(self, s1, s2):
            # Very basic similarity - just for fallback
            words1 = set(s1.split())
            words2 = set(s2.split())
            if not words1 or not words2:
                return 0
            intersection = words1.intersection(words2)
            return 100 * len(intersection) / max(len(words1), len(words2))
    
    fuzz = FallbackFuzz()

# ---------- Load sources (with categories)
def _load_sources() -> List[Dict[str, Any]]:
    path = os.getenv("NEWS_SOURCES_YAML", "trading_bot/news_sources.yaml")
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("sources", [])
    # fallback minimal
    return []

_SOURCES = _load_sources()

# ---------- Light caches
_CACHE: Dict[str, Tuple[float, Any]] = {}
def _get_cache(key: str, ttl: float):
    v = _CACHE.get(key)
    return (v[1] if v and (time.time() - v[0]) < ttl else None)

def _set_cache(key: str, data: Any): 
    _CACHE[key] = (time.time(), data)

# ---------- Lexicons & heuristics
FINANCE_TERMS = re.compile(r"\b(Fed|CPI|PPI|NFP|EPS|EBITDA|earnings|guidance|dividend|buyback|SEC|merger|acquisition|IPO|FOMC|Treasury|yield|bond|WTI|Brent|OPEC|tariff|GDP|PMI|ISM|recession|inflation)\b", re.I)
TICKER = re.compile(r"\$[A-Z]{1,5}\b")
LEFT_LEX = {"disenfranchised", "inequality", "climate crisis", "gun violence", "social justice", "workers", "union", "reproductive rights"}
RIGHT_LEX = {"woke", "illegal immigration", "border crisis", "tax burden", "regulation", "second amendment", "cancel culture", "law and order"}
HYPE = {"shocking", "explosive", "destroyed", "slams", "torches", "obliterates", "furious", "outrage", "disaster"}

def _canon_url(u: str) -> str:
    try:
        p = urlparse(u)
        clean_q = ""  # drop tracking
        return urlunparse((p.scheme, p.netloc, p.path, "", clean_q, ""))
    except:
        return u

async def _http_get(url: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=10, follow_redirects=True) as x:
            r = await x.get(url)
        return r.text
    except Exception:
        return ""

async def _read_feed(rss_url: str, limit: int) -> List[Dict[str, str]]:
    key = f"rss::{rss_url}"
    cached = _get_cache(key, 60)
    if cached is not None:
        return cached
    
    out = []
    if feedparser:
        feed = feedparser.parse(rss_url)
        for e in feed.entries[:limit * 2]:
            out.append({
                "title": e.get("title", ""),
                "link": e.get("link", ""),
                "published": e.get("published", "")
            })
    else:
        xml = await _http_get(rss_url)
        for m in re.finditer(r"<item>(.*?)</item>", xml, re.S | re.I):
            chunk = m.group(1)
            t = re.search(r"<title>(.*?)</title>", chunk, re.S | re.I)
            l = re.search(r"<link>(.*?)</link>", chunk, re.S | re.I)
            p = re.search(r"<pubDate>(.*?)</pubDate>", chunk, re.S | re.I)
            out.append({
                "title": (t.group(1) if t else "").strip(),
                "link": (l.group(1) if l else "").strip(),
                "published": (p.group(1) if p else "").strip()
            })
            if len(out) >= limit * 2:
                break
    
    _set_cache(key, out)
    return out

async def _extract_text(url: str) -> str:
    html = await _http_get(url)
    if trafi_extract:
        try:
            return trafi_extract(html) or ""
        except Exception:
            pass
    return re.sub(r"<[^>]+>", " ", html)[:12000]

def _info_score(text: str, prior_rel: float) -> float:
    if not text:
        return prior_rel * 0.6
    
    digits = sum(c.isdigit() for c in text)
    uppers = len(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", text))
    quotes = text.count('"') + text.count('"')
    hype = sum(1 for w in HYPE if w in text.lower())
    
    raw = (0.3 * min(digits / 60, 1) + 0.3 * min(uppers / 90, 1) + 0.2 * min(quotes / 8, 1)) - 0.2 * min(hype / 6, 1)
    return max(0.0, min(1.0, 0.6 * prior_rel + 0.4 * raw))

def _partisan_score(text: str, prior: float) -> float:
    t = text.lower()
    l = sum(1 for w in LEFT_LEX if w in t)
    r = sum(1 for w in RIGHT_LEX if w in t)
    raw = (r - l) / max(1, (r + l))
    return 0.6 * prior + 0.4 * raw

def _finance_score(title: str, text: str) -> float:
    t = f"{title}\n{text}"
    term = 1.0 if FINANCE_TERMS.search(t) else 0.0
    tick = min(3, len(TICKER.findall(t))) / 3.0
    return 0.6 * term + 0.4 * tick

def _sentiment(text: str) -> float:
    if _VADER:
        vs = _VADER.polarity_scores(text or "")
        return float(vs["compound"])
    
    # fallback: tiny lexicon
    pos = len(re.findall(r"\b(beat|surge|rally|growth|record|strong|bullish|raise|exceed|upgrade)\b", text, re.I))
    neg = len(re.findall(r"\b(miss|plunge|slump|decline|weak|bearish|cut|downgrade|lawsuit|sanction)\b", text, re.I))
    return (pos - neg) / max(1, (pos + neg))

def _cluster_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()

def _cluster(arts: List[Dict[str, Any]], threshold: int = 85) -> List[Dict[str, Any]]:
    clusters: List[List[Dict[str, Any]]] = []
    
    for a in arts:
        placed = False
        for cl in clusters:
            if fuzz.token_set_ratio(_cluster_key(a["title"]), _cluster_key(cl[0]["title"])) >= threshold:
                cl.append(a)
                placed = True
                break
        
        if not placed:
            clusters.append([a])
    
    out = []
    for cl in clusters:
        # aggregate
        sents = [x["sentiment"] for x in cl]
        parts = [x["partisan_score"] for x in cl]
        infos = [x["info_score"] for x in cl]
        fins = [x["finance_score"] for x in cl]
        top = max(cl, key=lambda x: len(x.get("title", "")))
        
        out.append({
            "headline": top["title"],
            "url": top["url"],
            "sentiment": sum(sents) / len(sents),
            "partisan_spread": (max(parts) - min(parts)) if parts else 0.0,
            "informational": sum(infos) / len(infos),
            "finance": max(fins) if fins else 0.0,
            "articles": cl,
            "sources": sorted({x["source"] for x in cl})
        })
    
    # sort by finance → info → |sentiment|
    out.sort(key=lambda c: (c["finance"], c["informational"], abs(c["sentiment"])), reverse=True)
    return out

async def sentiment_by_category(category: str, query: str = "", per_source: int = 5) -> Dict[str, Any]:
    cat = category.lower().strip()
    sources = [s for s in _SOURCES if s.get("category", "").lower() == cat] or _SOURCES
    arts: List[Dict[str, Any]] = []

    for s in sources:
        try:
            items = await _read_feed(s["rss"], per_source)
            for e in items:
                title = e["title"]
                link = e["link"]
                pub = e["published"]
                
                if query and query.lower() not in title.lower():
                    continue
                
                txt = await _extract_text(link)
                art = {
                    "source": s["name"],
                    "domain": s["domain"],
                    "title": title,
                    "url": _canon_url(link),
                    "published": pub,
                    "info_score": _info_score(txt, s.get("reliability_prior", 0.5)),
                    "partisan_score": _partisan_score(txt, s.get("bias_prior", 0.0)),
                    "finance_score": _finance_score(title, txt),
                    "sentiment": _sentiment(txt),
                }
                arts.append(art)
        except Exception as e:
            print(f"Error processing feed {s['name']}: {e}")
            continue

    clusters = _cluster(arts)
    
    # outlet rollups
    outlet: Dict[str, Any] = {}
    for a in arts:
        o = outlet.setdefault(a["source"], {"count": 0, "avg_sent": 0.0, "avg_partisan": 0.0, "avg_info": 0.0})
        o["count"] += 1
        o["avg_sent"] += (a["sentiment"] - o["avg_sent"]) / o["count"]
        o["avg_partisan"] += (a["partisan_score"] - o["avg_partisan"]) / o["count"]
        o["avg_info"] += (a["info_score"] - o["avg_info"]) / o["count"]

    return {"category": cat, "clusters": clusters, "outlets": outlet}

# --- HEALTH SHIM ---
from time import perf_counter

async def health() -> Dict[str, Any]:
    """
    Check multiple configured RSS providers; success if at least one is up.
    Safe for /health. Does not require API secrets.
    """
    res: Dict[str, Any] = {"service": "news", "ok": False, "providers": {}, "summary": {"up": 0, "total": 0}}
    ok = 0
    total = 0

    async def check_feed(name: str, rss_url: str, limit: int = 3) -> bool:
        start = perf_counter()
        try:
            items = await _read_feed(rss_url, limit)
            res["providers"][name] = {
                "ok": True,
                "n": len(items or []),
                "ms": int((perf_counter() - start) * 1000),
            }
            return True
        except Exception as e:
            res["providers"][name] = {"ok": False, "error": str(e)}
            return False

    # Use configured sources as providers (cap to first few for speed)
    for src in _SOURCES[:5]:
        total += 1
        name = src.get("name") or src.get("domain") or f"src{total}"
        ok += await check_feed(name, src.get("rss", ""), limit=3)

    res["ok"] = ok > 0
    res["summary"] = {"up": ok, "total": total}
    return res
