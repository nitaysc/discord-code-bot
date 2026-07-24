import asyncio
import io
import os
import random
import re
import sqlite3
import tempfile
import textwrap
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timedelta, timezone
from urllib.parse import quote, urlparse

SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
def _is_supported_image_url(url: str) -> bool:
    ext = os.path.splitext(urlparse(url).path.lower())[1]
    return ext in SUPPORTED_IMAGE_EXTS or not ext

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from openai import OpenAI
import wavelink
from minecraft_bot import MCBotManager, HAS_MC_BOT
from ddgs import DDGS
from PIL import Image, ImageDraw, ImageFont

try:
    from discord.ext import voice_recv as _voice_recv
    from discord.ext.voice_recv.extras.speechrecognition import SpeechRecognitionSink
    # Monkey-patch voice_recv router to catch OpusError (known library bug with DAVE/cameras)
    import discord.opus
    _original_do_run = _voice_recv.router.PacketRouter._do_run
    def _patched_do_run(self):
        try:
            _original_do_run(self)
        except discord.opus.OpusError as e:
            import logging
            logging.getLogger('discord.ext.voice_recv.router').warning(f"OpusError in router (non-fatal): {e}")
        except Exception as e:
            import logging
            logging.getLogger('discord.ext.voice_recv.router').warning(f"Router error (non-fatal): {e}")
    _voice_recv.router.PacketRouter._do_run = _patched_do_run
    HAS_VOICE_RECV = True
except ImportError:
    HAS_VOICE_RECV = False
    _voice_recv = None
    SpeechRecognitionSink = None

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
CLOUDFLARE_KEY = os.getenv("CLOUDFLARE_API_KEY")
CLOUDFLARE_ACCOUNT = os.getenv("CLOUDFLARE_ACCOUNT_ID")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
FREETHEAI_KEY = os.getenv("FREETHEAI_API_KEY")
HENRIKDEV_KEY = os.getenv("HENRIKDEV_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY")
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
TOGETHER_KEY = os.getenv("TOGETHER_API_KEY")

if not TOKEN:
    raise RuntimeError("Missing DISCORD_TOKEN in .env file")

# Provider configs — tried in priority order, skips capped ones
_PROVIDERS = []
_PROVIDER_DAILY_LIMITS = {}
_PROVIDER_USAGE = {}
_PROVIDER_RESET_DATE = {}


def _register_provider(name, base_url, api_key, text_model, vision_model, daily_limit=None):
    _PROVIDERS.append({
        "name": name,
        "client": OpenAI(base_url=base_url, api_key=api_key, timeout=5),
        "text_model": text_model,
        "vision_model": vision_model,
        "daily_limit": daily_limit,
    })
    if daily_limit:
        _PROVIDER_DAILY_LIMITS[name] = daily_limit
        _PROVIDER_USAGE[name] = 0
        _PROVIDER_RESET_DATE[name] = datetime.now(timezone.utc).date()


if GITHUB_TOKEN:
    _register_provider("GitHub (gpt-4o-mini)", "https://models.inference.ai.azure.com", GITHUB_TOKEN, "gpt-4o-mini", "gpt-4o-mini", 150)
    _register_provider("GitHub (gpt-4o)", "https://models.inference.ai.azure.com", GITHUB_TOKEN, "gpt-4o", "gpt-4o", 50)
if FREETHEAI_KEY:
    _register_provider("FreeTheAi", "https://api.freetheai.xyz/v1", FREETHEAI_KEY, "opc/deepseek-v4-flash-free", "kai/openrouter/free", 250)
if OPENROUTER_KEY:
    _register_provider("OpenRouter", "https://openrouter.ai/api/v1", OPENROUTER_KEY, "openrouter/free", "openrouter/free")
if HF_TOKEN:
    _register_provider("HuggingFace", "https://router.huggingface.co/v1", HF_TOKEN, "microsoft/phi-4", "microsoft/phi-4")
if CLOUDFLARE_KEY and CLOUDFLARE_ACCOUNT:
    _register_provider("Cloudflare", f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT}/ai/v1/", CLOUDFLARE_KEY, "@cf/meta/llama-3.1-8b-instruct", "@cf/meta/llama-3.1-8b-instruct")
if GROQ_KEY:
    _register_provider("Groq", "https://api.groq.com/openai/v1", GROQ_KEY, "llama-3.3-70b-versatile", "llama-3.2-11b-vision-preview")
if GEMINI_KEY:
    _register_provider("Gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", GEMINI_KEY, "gemini-2.0-flash", "gemini-2.0-flash")
if DEEPSEEK_KEY:
    _register_provider("DeepSeek", "https://api.deepseek.com", DEEPSEEK_KEY, "deepseek-chat", "deepseek-chat")
if MISTRAL_KEY:
    _register_provider("Mistral", "https://api.mistral.ai/v1", MISTRAL_KEY, "mistral-small-latest", "mistral-small-latest")
if TOGETHER_KEY:
    _register_provider("Together", "https://api.together.xyz/v1", TOGETHER_KEY, "meta-llama/Llama-3.3-70B-Instruct-Turbo", "meta-llama/Llama-3.3-70B-Instruct-Turbo")
if os.getenv("OPENAI_API_KEY"):
    _register_provider("OpenAI", "https://api.openai.com/v1", os.getenv("OPENAI_API_KEY"), "gpt-4o-mini", "gpt-4o-mini")

if not _PROVIDERS:
    raise RuntimeError("No AI API keys found — set at least one of: GITHUB_TOKEN, FREETHEAI_API_KEY, OPENROUTER_API_KEY, HF_TOKEN, etc.")

# Apply env var overrides to highest-priority provider
env_model = os.getenv("AI_MODEL")
env_vision = os.getenv("VISION_MODEL")
if env_model:
    _PROVIDERS[0]["text_model"] = env_model
if env_vision:
    _PROVIDERS[0]["vision_model"] = env_vision

provider_names = [p["name"] for p in _PROVIDERS]
names_with_limits = [f"{p['name']}({_PROVIDER_DAILY_LIMITS.get(p['name'], 'unlimited')}/day)" for p in _PROVIDERS]
print(f"[STARTUP] AI providers (priority): {' -> '.join(provider_names)}")
print(f"[STARTUP] Provider limits: {', '.join(names_with_limits)}")
print(f"[STARTUP] Search keys detected: SerpApi={'yes' if os.getenv('SERPAPI_API_KEY') else 'no'}, Bing={'yes' if os.getenv('BING_API_KEY') else 'no'}, Brave={'yes' if os.getenv('BRAVE_API_KEY') else 'no'}")


def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    last_err = None
    for attempt in range(3):
        try:
            results = list(DDGS().text(query, max_results=max_results))
            if not results:
                results = list(DDGS().news(query, max_results=max_results))
            if results:
                return results
        except Exception as e:
            last_err = e
            print(f"DuckDuckGo attempt {attempt+1} error: {e}")
        import time; time.sleep(1)
    print(f"DuckDuckGo search failed after retries: {last_err}")
    return []


def _search_images_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    last_err = None
    for attempt in range(3):
        try:
            results = list(DDGS().images(query, max_results=max_results))
            if results:
                return results
        except Exception as e:
            last_err = e
            print(f"DuckDuckGo image search attempt {attempt+1} error: {e}")
        import time; time.sleep(1)
    print(f"DuckDuckGo image search failed after retries: {last_err}")
    return []


async def _search_serpapi(query: str, api_key: str, max_results: int = 5) -> list[dict]:
    try:
        url = "https://serpapi.com/search.json"
        params = {"q": query, "api_key": api_key, "engine": "google", "num": max_results}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                data = await resp.json()
        results = []
        for r in data.get("organic_results", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "body": r.get("snippet", ""),
                "href": r.get("link", ""),
            })
        return results
    except Exception as e:
        print(f"SerpApi error: {e}")
        return []


async def _search_bing(query: str, api_key: str, max_results: int = 5) -> list[dict]:
    try:
        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        params = {"q": query, "count": max_results}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                data = await resp.json()
        results = []
        for r in data.get("webPages", {}).get("value", [])[:max_results]:
            results.append({
                "title": r.get("name", ""),
                "body": r.get("snippet", ""),
                "href": r.get("url", ""),
            })
        return results
    except Exception as e:
        print(f"Bing search error: {e}")
        return []


async def _search_brave(query: str, api_key: str, max_results: int = 5) -> list[dict]:
    try:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
        params = {"q": query, "count": max_results}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                data = await resp.json()
        results = []
        for r in data.get("web", {}).get("results", [])[:max_results]:
            results.append({
                "title": r.get("title", ""),
                "body": r.get("description", ""),
                "href": r.get("url", ""),
            })
        return results
    except Exception as e:
        print(f"Brave search error: {e}")
        return []


async def web_search(query: str, max_results: int = 5) -> str:
    results: list[dict] = []
    used = ""

    serp_key = os.getenv("SERPAPI_API_KEY")
    if serp_key and not results:
        results = await _search_serpapi(query, serp_key, max_results)
        used = "SerpApi"

    bing_key = os.getenv("BING_API_KEY")
    if bing_key and not results:
        results = await _search_bing(query, bing_key, max_results)
        used = "Bing"

    brave_key = os.getenv("BRAVE_API_KEY")
    if brave_key and not results:
        results = await _search_brave(query, brave_key, max_results)
        used = "Brave"

    if not results:
        results = await asyncio.to_thread(lambda: _search_duckduckgo(query, max_results))
        used = "DuckDuckGo"

    print(f"[SEARCH] backend={used or 'none'} query='{query}' results={len(results)}")

    if not results:
        return ""
    lines = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "")
        snippet = r.get("body", "")
        link = r.get("href", "")
        lines.append(f"[{i}] {title}\n{snippet}\n{link}")
    return "\n\n".join(lines)


async def web_image_search(query: str, max_results: int = 5) -> str:
    results = await asyncio.to_thread(lambda: _search_images_duckduckgo(query, max_results))
    if not results:
        return ""
    lines = []
    for i, r in enumerate(results[:max_results], 1):
        title = r.get("title", "")
        img_url = r.get("image", "")
        src_url = r.get("url", "")
        lines.append(f"[{i}] {title}\n{img_url}\n{src_url}")
    return "\n\n".join(lines)


async def _fetch_valorant_json(url: str) -> dict | None:
    headers = {}
    params = {}
    if HENRIKDEV_KEY:
        # Henrik supports both header and query param auth
        headers["Authorization"] = HENRIKDEV_KEY
        params["api_key"] = HENRIKDEV_KEY
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            async with session.get(url, headers=headers, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                print(f"[VALORANT] HTTP {resp.status} for {url}")
                return None
    except Exception as e:
        print(f"[VALORANT] error: {e}")
        return None


async def get_valorant_account(name: str, tag: str) -> dict | None:
    url = f"https://api.henrikdev.xyz/valorant/v2/account/{quote(name, safe='')}/{quote(tag, safe='')}"
    data = await _fetch_valorant_json(url)
    if not data:
        return None
    return data.get("data")


async def get_valorant_mmr(name: str, tag: str, region: str = "eu") -> dict | None:
    url = f"https://api.henrikdev.xyz/valorant/v3/mmr/{region}/pc/{quote(name, safe='')}/{quote(tag, safe='')}"
    data = await _fetch_valorant_json(url)
    if not data:
        return None
    return data.get("data")


async def get_valorant_matches(name: str, tag: str, region: str = "eu", limit: int = 3) -> list[dict]:
    url = f"https://api.henrikdev.xyz/valorant/v4/matches/{region}/pc/{quote(name, safe='')}/{quote(tag, safe='')}"
    data = await _fetch_valorant_json(url)
    if not data:
        return []
    matches = data.get("data", [])
    return matches[:limit]


def _cache_valorant_profile(name: str, tag: str, region: str | None = None, account_level: int | None = None):
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO valorant_profiles (name, tag, region, account_level, last_seen)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(name, tag) DO UPDATE SET
                 region=excluded.region,
                 account_level=excluded.account_level,
                 last_seen=excluded.last_seen""",
            (name, tag, region, account_level, datetime.now(timezone.utc).timestamp()),
        )
        conn.commit()
    except Exception as e:
        print(f"[VALORANT CACHE] error caching profile: {e}")


def _search_valorant_cache(name: str) -> list[tuple[str, str]]:
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT name, tag FROM valorant_profiles WHERE LOWER(name) LIKE LOWER(?) ORDER BY last_seen DESC LIMIT 25",
            (f"%{name}%",),
        )
        return [(row[0], row[1]) for row in cur.fetchall()]
    except Exception as e:
        print(f"[VALORANT CACHE] error searching cache: {e}")
        return []


_URL_CACHE: dict[str, tuple[str, float]] = {}
_URL_CACHE_TTL = 300  # 5 minutes


async def fetch_webpage_text(url: str) -> str | None:
    cached = _URL_CACHE.get(url)
    if cached and time.time() - cached[1] < _URL_CACHE_TTL:
        return cached[0]
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers={"User-Agent": "Mozilla/5.0"}, ssl=False, max_redirects=3) as resp:
                if resp.status != 200:
                    return None
                content_type = resp.headers.get("Content-Type", "")
                if "text/" not in content_type and "html" not in content_type:
                    return None
                text = await resp.text(errors="replace")
        # Strip HTML tags, extract readable text
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'https?://\S+', '', text)  # remove embedded URLs
        if len(text) > 3000:
            text = text[:3000] + "..."
        if len(text) < 50:
            return None  # too short, probably not useful
        _URL_CACHE[url] = (text, time.time())
        return text
    except Exception as e:
        print(f"[URL FETCH] error fetching {url}: {e}")
        return None


def _save_user_note(user_id: int, key: str, value: str):
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO user_notes (user_id, note_key, note_value, updated_at) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(user_id, note_key) DO UPDATE SET note_value = excluded.note_value, updated_at = excluded.updated_at",
            (user_id, key, value, time.time()),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[NOTES] error saving note: {e}")


def _get_user_notes(user_id: int) -> str:
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT note_key, note_value FROM user_notes WHERE user_id = ? ORDER BY updated_at DESC", (user_id,))
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return ""
        notes = [f"{k}: {v}" for k, v in rows]
        return "User's saved notes:\n" + "\n".join(notes)
    except Exception as e:
        print(f"[NOTES] error reading notes: {e}")
        return ""


VALORANT_REGIONS = ["na", "eu", "ap", "kr", "latam", "br"]


def _extract_valorant_profiles_from_search(text: str, query_name: str) -> list[tuple[str, str]]:
    """Extract (name, tag) pairs from tracker.gg search result snippets/links."""
    found: list[tuple[str, str]] = []
    seen = set()
    # tracker.gg profile URLs look like: .../tracker.gg/valorant/profile/riot/{name}%23{tag}/overview
    pattern = re.compile(r"/valorant/profile/riot/([^/\s%]+)(?:%23|#)([^/\s%]+)/", re.IGNORECASE)
    for m in pattern.finditer(text):
        name_part = m.group(1).replace("%20", " ").replace("+", " ")
        tag_part = m.group(2)
        key = (name_part.lower(), tag_part.lower())
        if key not in seen:
            seen.add(key)
            found.append((name_part, tag_part))
    # Fallback: look for "Name #tag" or "Name#tag" anywhere in text
    if not found:
        fallback = re.compile(rf"({re.escape(query_name)}[^#\n]{{0,15}}?)\s*#(\w+)", re.IGNORECASE)
        for m in fallback.finditer(text):
            name_part = m.group(1).strip()
            tag_part = m.group(2)
            key = (name_part.lower(), tag_part.lower())
            if key not in seen:
                seen.add(key)
                found.append((name_part, tag_part))
    return found[:10]


async def search_valorant_profiles(name: str) -> list[tuple[str, str]]:
    """Search for Valorant profiles using multiple web queries and combine results."""
    queries = [
        f"site:tracker.gg/valorant/profile/riot \"{name}\"",
        f"site:tracker.gg/valorant/profile/riot {name}",
        f"tracker.gg valorant {name} profile",
        f"\"{name}\" valorant tracker",
    ]
    seen = set()
    found: list[tuple[str, str]] = []
    for query in queries:
        try:
            results_text = await web_search(query, max_results=10)
            for pname, ptag in _extract_valorant_profiles_from_search(results_text, name):
                key = (pname.lower(), ptag.lower())
                if key not in seen:
                    seen.add(key)
                    found.append((pname, ptag))
        except Exception as e:
            print(f"[VALORANT SEARCH] query failed '{query}': {e}")
    # If the user typed an exact name#tag and web search didn't find it,
    # include it as a candidate so they can still select it.
    if "#" in name:
        parts = name.rsplit("#", 1)
        exact_key = (parts[0].strip().lower(), parts[1].strip().lower())
        if exact_key not in seen:
            seen.add(exact_key)
            found.insert(0, (parts[0].strip(), parts[1].strip()))
    return found[:25]


class ValorantProfileSelect(discord.ui.Select):
    def __init__(self, profiles: list[tuple[str, str]]):
        options = []
        for i, (pname, ptag) in enumerate(profiles[:25]):
            label = f"{pname}#{ptag}"[:25]
            options.append(discord.SelectOption(label=label, value=f"{pname}|{ptag}", description=f"Select {pname}#{ptag}"[:50]))
        super().__init__(placeholder="Choose a player...", options=options)
        self.profiles = profiles

    async def callback(self, interaction: discord.Interaction):
        name, tag = self.values[0].split("|", 1)
        await interaction.response.defer()
        account = await get_valorant_account(name, tag)
        if not account:
            await interaction.followup.send(f":x: Couldn't load account for **{name}#{tag}**.")
            return

        region = "eu"
        region_hint = account.get("region", "")
        if region_hint and region_hint.lower() in VALORANT_REGIONS:
            region = region_hint.lower()
        mmr = await get_valorant_mmr(name, tag, region)

        card = account.get("card", {})
        embed = discord.Embed(
            title=f"{account.get('name', name)}#{account.get('tag', tag)}",
            color=0xfa4454,
        )
        if isinstance(card, dict):
            if card.get("small"):
                embed.set_thumbnail(url=card["small"])
            if card.get("wide"):
                embed.set_image(url=card["wide"])
        embed.add_field(name="Region", value=account.get("region", region).upper() or "Unknown", inline=True)
        embed.add_field(name="Account Level", value=str(account.get("account_level", "?")), inline=True)
        if mmr:
            rank_name = (
                mmr.get("currenttierpatched")
                or mmr.get("tier")
                or mmr.get("currenttier", "Unknown")
            )
            rr = mmr.get("ranking_in_tier") if "ranking_in_tier" in mmr else mmr.get("rr", "?")
            last_game = mmr.get("mmr_change_to_last_game") if "mmr_change_to_last_game" in mmr else mmr.get("last_rank_change", 0)
            last_game_text = f"{last_game:+d} RR last game" if isinstance(last_game, int) else ""
            embed.add_field(name="Rank", value=f"{rank_name} — {rr} RR\n{last_game_text}", inline=False)
            if isinstance(mmr.get("images"), dict) and mmr["images"].get("small"):
                embed.set_thumbnail(url=mmr["images"]["small"])
        else:
            embed.add_field(name="Rank", value="Could not load ranked data.", inline=False)

        await interaction.followup.send(embed=embed)


SEARCH_TRIGGER_WORDS = {
    "latest", "current", "today", "news", "weather", "price", "prices",
    "score", "scores", "update", "recent", "happened", "happening", "live",
    "stock", "crypto", "bitcoin", "election", "winner", "winners", "results",
    "release date", "release", "released", "age of", "net worth",
    "episode", "episodes", "season", "seasons", "premiere", "premieres",
    "schedule", "scheduled", "dropping", "drops", "drop", "upcoming",
    "future", "plan to", "planning", "planned", "announced",
    "search for", "look up", "look online", "find online", "check online", "google",
    "lookonline", "lookup", "searchonline",
    "how to",
    "world cup", "olympics", "super bowl", "champions league", "eurovision",
    "2025", "2026", "2027", "yesterday", "tomorrow", "this week", "last week",
    "coming out", "out now", "out today", "launch",
}

# Common bot command prefixes to strip before checking question starters
BOT_PREFIXES = (".", "!", "?", "/", ">", "-", "+", "$", "%", "^", "&", "*", "~")


def _clean_search_question(question: str) -> str:
    """Remove bot mentions, extra punctuation, and filler words."""
    q = re.sub(r"<@!?\d+>", "", question)
    q = re.sub(r"[@\s]+\bnull\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r"\bnull\b", "", q, flags=re.IGNORECASE)
    q = re.sub(r"[^\w\s'\-?]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def _strip_prefix(text: str) -> str:
    """Remove bot command prefix (e.g. .fxnitay) from the start."""
    for p in BOT_PREFIXES:
        if text.startswith(p):
            # Strip the prefix character and any following word (the command name)
            rest = text[len(p):].lstrip()
            # Remove the first word (command name like "fxnitay")
            parts = rest.split(None, 1)
            if len(parts) > 1:
                return parts[1]
            return rest
    return text


def should_search(question: str) -> tuple[bool, str]:
    lowered = question.lower()
    if any(re.search(r'\b' + re.escape(w) + r'\b', lowered) for w in SEARCH_TRIGGER_WORDS):
        return True, _clean_search_question(question)
    # Strip bot prefix and check question starters
    unprefixed = _strip_prefix(lowered)
    question_starters = (
        "when ", "where ", "who ", "how much ", "how many ", "how old ",
        "why is ", "why are ", "what is ", "whats ", "what's ",
        "what are ", "what was ", "what were ", "what does ", "what do ",
        "tell me when ", "tell me what ", "tell me who ", "tell me where ",
        "do you know ", "do you have ", "has there been ", "is there ",
        "are there ", "any news ", "any update ", "any info ",
    )
    if lowered.startswith(question_starters) or unprefixed.startswith(question_starters):
        return True, _clean_search_question(question)
    return False, ""


SEARCH_STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "cant", "cannot",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "there", "here",
    "and", "or", "but", "so", "yet", "for", "nor", "as", "of", "in", "on",
    "at", "to", "from", "by", "with", "about", "into", "through", "during",
    "before", "after", "above", "below", "between", "among", "within",
    "what", "whats", "what's", "which", "who", "whom", "whose", "where",
    "when", "why", "how", "hows", "how's", "howmuch", "howmuchis",
    "tell", "give", "show", "find", "search", "look", "check", "get",
    "please", "pls", "thank", "thanks", "bro", "dude", "hey", "hi", "hello",
    "made", "make", "like", "just", "only", "also", "even", "very", "really",
    "much", "many", "some", "any", "all", "both", "each", "every", "most",
    "other", "another", "such", "no", "not", "yes", "ok", "okay",
}


def _strip_filler_words(text: str) -> str:
    """Remove stop/filler words and collapse repeated letters."""
    words = []
    for w in text.lower().split():
        w = re.sub(r"(.)\1{2,}", r"\1\1", w)  # soooo -> soo
        w = w.strip("?.,!;:'\"/")
        if w and w not in SEARCH_STOP_WORDS:
            words.append(w)
    return " ".join(words)


def _extract_search_query(question: str) -> str:
    """Turn a messy question into a concise search query using deterministic cleanup."""
    clean = _clean_search_question(question)
    query = _strip_filler_words(clean)
    # Keep it short but preserve the most important words (brand/product/topic first)
    words = query.split()
    if len(words) > 10:
        # Simple heuristic: keep first 4 and last 2 words, drop middle filler
        query = " ".join(words[:4] + words[-2:])
    return query[:120]


LANG_EXTENSIONS = {
    "lua": ".lua", "luau": ".luau", "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js", "typescript": ".ts", "ts": ".ts",
    "html": ".html", "css": ".css", "c": ".c", "cpp": ".cpp", "c++": ".cpp",
    "csharp": ".cs", "cs": ".cs", "java": ".java", "go": ".go", "rust": ".rs",
    "ruby": ".rb", "php": ".php", "swift": ".swift", "kotlin": ".kt",
    "sql": ".sql", "shell": ".sh", "bash": ".sh", "powershell": ".ps1",
    "ps1": ".ps1", "bat": ".bat", "vbs": ".vbs", "ahk": ".ahk",
}

FILE_EXTENSIONS = {
    **LANG_EXTENSIONS,
    "json": ".json", "yaml": ".yml", "yml": ".yml", "xml": ".xml",
    "markdown": ".md", "md": ".md", "txt": ".txt", "csv": ".csv",
    "ini": ".ini", "cfg": ".cfg", "toml": ".toml", "conf": ".conf",
    "reg": ".reg", "log": ".log", "env": ".env", "gitignore": ".gitignore",
    "dockerfile": "", "makefile": "", "readme": ".md",
}

BINARY_TYPES = {"exe", "dll", "so", "dylib", "bin", "apk", "ipa", "msi",
                "app", "pkg", "deb", "rpm", "img", "iso", "zip", "rar", "7z",
                "png", "jpg", "jpeg", "gif", "bmp", "ico", "mp3", "mp4",
                "wav", "avi", "mkv", "pdf", "docx", "xlsx", "pptx", "ttf", "otf"}

CREATE_KEYWORDS = [
    "make", "create", "write", "generate", "build", "code", "script",
    "program", "file", "give me", "send me", "show me", "output",
]

MAX_HISTORY = 50
message_history: dict[int, deque[dict]] = {}

DATABASE_URL = os.getenv("DATABASE_URL")
DB_PATH = os.getenv("DB_PATH", os.path.join(os.path.dirname(__file__), "bot.db"))

USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    try:
        import psycopg2
        from psycopg2 import pool as psycopg2_pool
    except ImportError:
        psycopg2 = None
        psycopg2_pool = None
else:
    psycopg2 = None
    psycopg2_pool = None

_pg_pool = None
_pg_schema_initialized = False


def _get_pg_pool():
    global _pg_pool
    if _pg_pool is None and psycopg2_pool:
        _pg_pool = psycopg2_pool.ThreadedConnectionPool(2, 10, DATABASE_URL)
    return _pg_pool


class _PostgresCursor:
    def __init__(self, cur):
        self._cur = cur

    def execute(self, query, params=()):
        return self._cur.execute(query.replace("?", "%s"), params)

    def executemany(self, query, params_list):
        return self._cur.executemany(query.replace("?", "%s"), params_list)

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()


class _PostgresConnection:
    def __init__(self, conn, pool):
        self._conn = conn
        self._pool = pool

    def cursor(self):
        return _PostgresCursor(self._conn.cursor())

    def execute(self, query, params=()):
        cur = self._conn.cursor()
        cur.execute(query.replace("?", "%s"), params)
        return _PostgresCursor(cur)

    def executescript(self, script):
        cur = self._conn.cursor()
        statements = self._split_statements(script)
        for statement in statements:
            if statement:
                cur.execute(statement)
        cur.close()

    @staticmethod
    def _split_statements(script: str) -> list[str]:
        statements = []
        current = []
        in_dollar = False
        i = 0
        while i < len(script):
            if not in_dollar and script.startswith("$$", i):
                in_dollar = True
                current.append("$$")
                i += 2
                continue
            if in_dollar and script.startswith("$$", i):
                in_dollar = False
                current.append("$$")
                i += 2
                continue
            if script[i] == ";" and not in_dollar:
                statements.append("".join(current).strip())
                current = []
                i += 1
                continue
            current.append(script[i])
            i += 1
        remainder = "".join(current).strip()
        if remainder:
            statements.append(remainder)
        return statements

    def commit(self):
        self._conn.commit()

    def close(self):
        if self._pool:
            self._pool.putconn(self._conn)
        else:
            self._conn.close()


SCHEMA_SCRIPT_SQLITE = """
CREATE TABLE IF NOT EXISTS levels (
    guild_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    xp BIGINT DEFAULT 0,
    messages INTEGER DEFAULT 0,
    voice_minutes INTEGER DEFAULT 0,
    last_xp REAL DEFAULT 0,
    PRIMARY KEY (guild_id, user_id)
);
CREATE TABLE IF NOT EXISTS level_settings (
    guild_id BIGINT NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    PRIMARY KEY (guild_id, key)
);
CREATE TABLE IF NOT EXISTS role_rewards (
    guild_id BIGINT NOT NULL,
    level INTEGER NOT NULL,
    role_id BIGINT NOT NULL,
    PRIMARY KEY (guild_id, level)
);
CREATE TABLE IF NOT EXISTS xp_blacklist (
    guild_id BIGINT NOT NULL,
    target_id BIGINT NOT NULL,
    target_type TEXT NOT NULL,
    PRIMARY KEY (guild_id, target_id, target_type)
);
CREATE TABLE IF NOT EXISTS xp_multipliers (
    guild_id BIGINT NOT NULL,
    target_id BIGINT NOT NULL,
    target_type TEXT NOT NULL,
    multiplier REAL NOT NULL,
    PRIMARY KEY (guild_id, target_id, target_type)
);
CREATE TABLE IF NOT EXISTS ticket_settings (
    guild_id BIGINT PRIMARY KEY,
    category_id BIGINT,
    log_channel_id BIGINT,
    support_role_id BIGINT,
    welcome_message TEXT DEFAULT 'Support will be with you shortly.',
    transcript_enabled INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS ticket_panels (
    panel_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id BIGINT NOT NULL,
    channel_id BIGINT NOT NULL,
    message_id BIGINT NOT NULL,
    label TEXT NOT NULL,
    emoji TEXT,
    description TEXT,
    category_id BIGINT,
    support_role_id BIGINT
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
    guild_id BIGINT NOT NULL,
    channel_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    claimed_by BIGINT,
    status TEXT DEFAULT 'open',
    created_at REAL DEFAULT 0,
    closed_at REAL
);
CREATE TABLE IF NOT EXISTS valorant_profiles (
    name TEXT NOT NULL,
    tag TEXT NOT NULL,
    region TEXT,
    account_level INTEGER,
    last_seen REAL DEFAULT 0,
    PRIMARY KEY (name, tag)
);
CREATE TABLE IF NOT EXISTS temp_voice_creators (
    guild_id BIGINT NOT NULL,
    channel_id BIGINT NOT NULL,
    name_format TEXT DEFAULT '{OWNER_USERNAME}s Channel',
    user_limit INTEGER DEFAULT 0,
    privacy_mode TEXT DEFAULT 'public',
    category_id BIGINT,
    PRIMARY KEY (guild_id, channel_id)
);
CREATE TABLE IF NOT EXISTS temp_voice_active (
    channel_id BIGINT PRIMARY KEY,
    guild_id BIGINT NOT NULL,
    owner_id BIGINT NOT NULL,
    creator_id BIGINT NOT NULL,
    created_at REAL DEFAULT 0,
    name TEXT,
    user_limit INTEGER DEFAULT 0,
    privacy TEXT DEFAULT 'public'
);
CREATE TABLE IF NOT EXISTS temp_voice_prefs (
    guild_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    saved_name TEXT,
    saved_limit INTEGER,
    saved_privacy TEXT,
    PRIMARY KEY (guild_id, user_id)
);
CREATE TABLE IF NOT EXISTS user_notes (
    user_id BIGINT NOT NULL,
    note_key TEXT NOT NULL,
    note_value TEXT NOT NULL,
    updated_at REAL DEFAULT 0,
    PRIMARY KEY (user_id, note_key)
);
CREATE TABLE IF NOT EXISTS welcome_settings (
    guild_id BIGINT PRIMARY KEY,
    channel_id BIGINT,
    message TEXT DEFAULT 'Welcome {user} to **{server}**! You are member **#{count}**!',
    image_url TEXT,
    enabled INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS goodbye_settings (
    guild_id BIGINT PRIMARY KEY,
    channel_id BIGINT,
    message TEXT DEFAULT 'Goodbye {user.name}, we will miss you!',
    enabled INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS autoroles (
    guild_id BIGINT NOT NULL,
    role_id BIGINT NOT NULL,
    PRIMARY KEY (guild_id, role_id)
);
CREATE TABLE IF NOT EXISTS welcome_dm_settings (
    guild_id BIGINT PRIMARY KEY,
    enabled INTEGER DEFAULT 0,
    message TEXT DEFAULT 'Welcome to **{server}**! Check out the rules in #rules!'
);
CREATE TABLE IF NOT EXISTS counting_settings (
    guild_id BIGINT PRIMARY KEY,
    channel_id BIGINT
);
CREATE TABLE IF NOT EXISTS counting_stats (
    guild_id BIGINT PRIMARY KEY,
    current_count BIGINT DEFAULT 0,
    highest_count BIGINT DEFAULT 0,
    last_user_id BIGINT DEFAULT 0
);
"""
SCHEMA_SCRIPT_POSTGRES = """
CREATE TABLE IF NOT EXISTS levels (
    guild_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    xp BIGINT DEFAULT 0,
    messages INTEGER DEFAULT 0,
    voice_minutes INTEGER DEFAULT 0,
    last_xp REAL DEFAULT 0,
    PRIMARY KEY (guild_id, user_id)
);
CREATE TABLE IF NOT EXISTS level_settings (
    guild_id BIGINT NOT NULL,
    key TEXT NOT NULL,
    value TEXT,
    PRIMARY KEY (guild_id, key)
);
CREATE TABLE IF NOT EXISTS role_rewards (
    guild_id BIGINT NOT NULL,
    level INTEGER NOT NULL,
    role_id BIGINT NOT NULL,
    PRIMARY KEY (guild_id, level)
);
CREATE TABLE IF NOT EXISTS xp_blacklist (
    guild_id BIGINT NOT NULL,
    target_id BIGINT NOT NULL,
    target_type TEXT NOT NULL,
    PRIMARY KEY (guild_id, target_id, target_type)
);
CREATE TABLE IF NOT EXISTS xp_multipliers (
    guild_id BIGINT NOT NULL,
    target_id BIGINT NOT NULL,
    target_type TEXT NOT NULL,
    multiplier REAL NOT NULL,
    PRIMARY KEY (guild_id, target_id, target_type)
);
CREATE TABLE IF NOT EXISTS ticket_settings (
    guild_id BIGINT PRIMARY KEY,
    category_id BIGINT,
    log_channel_id BIGINT,
    support_role_id BIGINT,
    welcome_message TEXT DEFAULT 'Support will be with you shortly.',
    transcript_enabled INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS ticket_panels (
    panel_id SERIAL PRIMARY KEY,
    guild_id BIGINT NOT NULL,
    channel_id BIGINT NOT NULL,
    message_id BIGINT NOT NULL,
    label TEXT NOT NULL,
    emoji TEXT,
    description TEXT,
    category_id BIGINT,
    support_role_id BIGINT
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id SERIAL PRIMARY KEY,
    guild_id BIGINT NOT NULL,
    channel_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    claimed_by BIGINT,
    status TEXT DEFAULT 'open',
    created_at REAL DEFAULT 0,
    closed_at REAL
);
CREATE TABLE IF NOT EXISTS valorant_profiles (
    name TEXT NOT NULL,
    tag TEXT NOT NULL,
    region TEXT,
    account_level INTEGER,
    last_seen REAL DEFAULT 0,
    PRIMARY KEY (name, tag)
);
CREATE TABLE IF NOT EXISTS temp_voice_creators (
    guild_id BIGINT NOT NULL,
    channel_id BIGINT NOT NULL,
    name_format TEXT DEFAULT '{OWNER_USERNAME}s Channel',
    user_limit INTEGER DEFAULT 0,
    privacy_mode TEXT DEFAULT 'public',
    category_id BIGINT,
    PRIMARY KEY (guild_id, channel_id)
);
CREATE TABLE IF NOT EXISTS temp_voice_active (
    channel_id BIGINT PRIMARY KEY,
    guild_id BIGINT NOT NULL,
    owner_id BIGINT NOT NULL,
    creator_id BIGINT NOT NULL,
    created_at REAL DEFAULT 0,
    name TEXT,
    user_limit INTEGER DEFAULT 0,
    privacy TEXT DEFAULT 'public'
);
CREATE TABLE IF NOT EXISTS temp_voice_prefs (
    guild_id BIGINT NOT NULL,
    user_id BIGINT NOT NULL,
    saved_name TEXT,
    saved_limit INTEGER,
    saved_privacy TEXT,
    PRIMARY KEY (guild_id, user_id)
);
CREATE TABLE IF NOT EXISTS user_notes (
    user_id BIGINT NOT NULL,
    note_key TEXT NOT NULL,
    note_value TEXT NOT NULL,
    updated_at REAL DEFAULT 0,
    PRIMARY KEY (user_id, note_key)
);
CREATE TABLE IF NOT EXISTS welcome_settings (
    guild_id BIGINT PRIMARY KEY,
    channel_id BIGINT,
    message TEXT DEFAULT 'Welcome {user} to **{server}**! You are member **#{count}**!',
    image_url TEXT,
    enabled INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS goodbye_settings (
    guild_id BIGINT PRIMARY KEY,
    channel_id BIGINT,
    message TEXT DEFAULT 'Goodbye {user.name}, we will miss you!',
    enabled INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS autoroles (
    guild_id BIGINT NOT NULL,
    role_id BIGINT NOT NULL,
    PRIMARY KEY (guild_id, role_id)
);
CREATE TABLE IF NOT EXISTS welcome_dm_settings (
    guild_id BIGINT PRIMARY KEY,
    enabled INTEGER DEFAULT 0,
    message TEXT DEFAULT 'Welcome to **{server}**! Check out the rules in #rules!'
);
CREATE TABLE IF NOT EXISTS counting_settings (
    guild_id BIGINT PRIMARY KEY,
    channel_id BIGINT
);
CREATE TABLE IF NOT EXISTS counting_stats (
    guild_id BIGINT PRIMARY KEY,
    current_count BIGINT DEFAULT 0,
    highest_count BIGINT DEFAULT 0,
    last_user_id BIGINT DEFAULT 0
);
"""

PG_MIGRATION_SCRIPT = """
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='levels' AND column_name='guild_id' AND data_type='integer') THEN
        ALTER TABLE levels ALTER COLUMN guild_id TYPE BIGINT;
        ALTER TABLE levels ALTER COLUMN user_id TYPE BIGINT;
        ALTER TABLE levels ALTER COLUMN xp TYPE BIGINT;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='level_settings' AND column_name='guild_id' AND data_type='integer') THEN
        ALTER TABLE level_settings ALTER COLUMN guild_id TYPE BIGINT;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='role_rewards' AND column_name='guild_id' AND data_type='integer') THEN
        ALTER TABLE role_rewards ALTER COLUMN guild_id TYPE BIGINT;
        ALTER TABLE role_rewards ALTER COLUMN role_id TYPE BIGINT;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='xp_blacklist' AND column_name='guild_id' AND data_type='integer') THEN
        ALTER TABLE xp_blacklist ALTER COLUMN guild_id TYPE BIGINT;
        ALTER TABLE xp_blacklist ALTER COLUMN target_id TYPE BIGINT;
    END IF;
    IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='xp_multipliers' AND column_name='guild_id' AND data_type='integer') THEN
        ALTER TABLE xp_multipliers ALTER COLUMN guild_id TYPE BIGINT;
        ALTER TABLE xp_multipliers ALTER COLUMN target_id TYPE BIGINT;
    END IF;
END $$;
"""


def get_db():
    global _pg_schema_initialized
    if USE_POSTGRES and psycopg2:
        pg_pool = _get_pg_pool()
        if pg_pool:
            conn = pg_pool.getconn()
            wrapper = _PostgresConnection(conn, pg_pool)
            if not _pg_schema_initialized:
                wrapper.executescript(SCHEMA_SCRIPT_POSTGRES)
                wrapper.executescript(PG_MIGRATION_SCRIPT)
                wrapper.commit()
                _pg_schema_initialized = True
            return wrapper
        conn = psycopg2.connect(DATABASE_URL)
        wrapper = _PostgresConnection(conn, None)
        if not _pg_schema_initialized:
            wrapper.executescript(SCHEMA_SCRIPT_POSTGRES)
            wrapper.executescript(PG_MIGRATION_SCRIPT)
            wrapper.commit()
            _pg_schema_initialized = True
        return wrapper
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA_SCRIPT_SQLITE)
    conn.commit()
    return conn


def _get_setting(guild_id: int, key: str, default=None):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT value FROM level_settings WHERE guild_id = ? AND key = ?",
        (guild_id, key),
    )
    row = cur.fetchone()
    conn.close()
    return row[0] if row else default


def _set_setting(guild_id: int, key: str, value):
    conn = get_db()
    conn.execute(
        "INSERT INTO level_settings (guild_id, key, value) VALUES (?, ?, ?) "
        "ON CONFLICT(guild_id, key) DO UPDATE SET value = excluded.value",
        (guild_id, key, str(value)),
    )
    conn.commit()
    conn.close()


def xp_for_level(level: int) -> int:
    return int(100 * (level ** 1.8))


def level_from_xp(xp: int) -> int:
    level = 0
    while xp_for_level(level + 1) <= xp:
        level += 1
    return level


def _is_blacklisted(guild_id: int, channel_id: int, role_ids: list[int]) -> bool:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM xp_blacklist WHERE guild_id = ? AND target_id = ? AND target_type = 'channel'",
        (guild_id, channel_id),
    )
    if cur.fetchone():
        conn.close()
        return True
    if role_ids:
        placeholders = ",".join("?" * len(role_ids))
        cur.execute(
            f"SELECT 1 FROM xp_blacklist WHERE guild_id = ? AND target_id IN ({placeholders}) AND target_type = 'role' LIMIT 1",
            (guild_id, *role_ids),
        )
        if cur.fetchone():
            conn.close()
            return True
    conn.close()
    return False


def _get_multiplier(guild_id: int, channel_id: int, role_ids: list[int]) -> float:
    conn = get_db()
    cur = conn.cursor()
    mult = 1.0
    cur.execute(
        "SELECT multiplier FROM xp_multipliers WHERE guild_id = ? AND target_id = ? AND target_type = 'channel'",
        (guild_id, channel_id),
    )
    row = cur.fetchone()
    if row:
        mult *= row[0]
    if role_ids:
        placeholders = ",".join("?" * len(role_ids))
        cur.execute(
            f"SELECT multiplier FROM xp_multipliers WHERE guild_id = ? AND target_id IN ({placeholders}) AND target_type = 'role'",
            (guild_id, *role_ids),
        )
        for row in cur.fetchall():
            mult *= row[0]
    conn.close()
    return mult


_xp_cooldowns: dict[tuple[int, int], float] = {}


def add_message_xp(guild_id: int, user_id: int, channel_id: int, role_ids: list[int]) -> tuple[int, int, bool] | None:
    if _is_blacklisted(guild_id, channel_id, role_ids):
        print(f"[XP] blacklisted: guild={guild_id} user={user_id} channel={channel_id}")
        return None
    now = datetime.now(timezone.utc).timestamp()
    key = (guild_id, user_id)
    cooldown = float(_get_setting(guild_id, "cooldown", 30))
    if key in _xp_cooldowns and now - _xp_cooldowns[key] < cooldown:
        print(f"[XP] cooldown: guild={guild_id} user={user_id} remaining={cooldown - (now - _xp_cooldowns[key]):.1f}s")
        return None
    _xp_cooldowns[key] = now
    min_xp = int(_get_setting(guild_id, "min_xp", 15))
    max_xp = int(_get_setting(guild_id, "max_xp", 25))
    gained = random.randint(min_xp, max_xp)
    mult = _get_multiplier(guild_id, channel_id, role_ids)
    gained = int(gained * mult)
    print(f"[XP] adding: guild={guild_id} user={user_id} amount={gained}")
    return _apply_xp(guild_id, user_id, gained, "message")


def add_voice_xp(guild_id: int, user_id: int, amount: int) -> tuple[int, int, bool] | None:
    return _apply_xp(guild_id, user_id, amount, "voice")


def _apply_xp(guild_id: int, user_id: int, amount: int, source: str) -> tuple[int, int, bool]:
    now = datetime.now(timezone.utc).timestamp()
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT xp FROM levels WHERE guild_id = ? AND user_id = ?",
        (guild_id, user_id),
    )
    row = cur.fetchone()
    old_xp = row[0] if row else 0
    new_xp = old_xp + amount
    old_level = level_from_xp(old_xp)
    new_level = level_from_xp(new_xp)
    leveled_up = new_level > old_level
    if source == "message":
        cur.execute(
            """INSERT INTO levels (guild_id, user_id, xp, messages, last_xp)
               VALUES (?, ?, ?, 1, ?)
               ON CONFLICT(guild_id, user_id) DO UPDATE SET
                 xp = levels.xp + ?,
                 messages = levels.messages + 1,
                 last_xp = ?""",
            (guild_id, user_id, new_xp, now, amount, now),
        )
    else:
        cur.execute(
            """INSERT INTO levels (guild_id, user_id, xp, voice_minutes, last_xp)
               VALUES (?, ?, ?, 1, ?)
               ON CONFLICT(guild_id, user_id) DO UPDATE SET
                 xp = levels.xp + ?,
                 voice_minutes = levels.voice_minutes + 1,
                 last_xp = ?""",
            (guild_id, user_id, new_xp, now, amount, now),
        )
    conn.commit()
    conn.close()
    print(f"[XP] saved: guild={guild_id} user={user_id} xp={new_xp} level={new_level} leveled_up={leveled_up}")
    return new_xp, new_level, leveled_up


async def handle_level_up(member: discord.Member, channel: discord.TextChannel, level: int):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT role_id FROM role_rewards WHERE guild_id = ? AND level = ?",
        (member.guild.id, level),
    )
    rows = cur.fetchall()
    conn.close()
    roles_added = []
    for (role_id,) in rows:
        role = member.guild.get_role(role_id)
        if role and role not in member.roles:
            try:
                await member.add_roles(role)
                roles_added.append(role.mention)
            except Exception:
                pass
    enabled = (await asyncio.to_thread(_get_setting, member.guild.id, "levelup_enabled", "true")).lower()
    if enabled != "true":
        return
    levelup_channel_id = await asyncio.to_thread(_get_setting, member.guild.id, "levelup_channel")
    target_channel = channel
    if levelup_channel_id:
        ch = member.guild.get_channel(int(levelup_channel_id))
        if ch:
            target_channel = ch
    msg = f":tada: GG {member.mention}, you reached **Level {level}**!"
    if roles_added:
        msg += f"\n:medal: Reward: {', '.join(roles_added)}"
    try:
        await target_channel.send(msg, delete_after=15)
    except Exception:
        pass


def get_rank(guild_id: int, user_id: int) -> dict | None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT xp, messages, voice_minutes FROM levels WHERE guild_id = ? AND user_id = ?",
        (guild_id, user_id),
    )
    row = cur.fetchone()
    if not row:
        conn.close()
        return None
    xp, messages, voice_minutes = row
    cur.execute(
        "SELECT COUNT(*) FROM levels WHERE guild_id = ? AND xp > ?",
        (guild_id, xp),
    )
    rank = cur.fetchone()[0] + 1
    conn.close()
    return {
        "xp": xp,
        "level": level_from_xp(xp),
        "rank": rank,
        "messages": messages,
        "voice_minutes": voice_minutes,
    }


def get_leaderboard(guild_id: int, limit: int = 10) -> list[tuple[int, int, int]]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT user_id, xp FROM levels WHERE guild_id = ? ORDER BY xp DESC LIMIT ?",
        (guild_id, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [(user_id, xp, level_from_xp(xp)) for user_id, xp in rows]


# --- Ticket system ---

TICKET_SETTINGS_COLUMNS = {
    "category_id": "BIGINT",
    "log_channel_id": "BIGINT",
    "support_role_id": "BIGINT",
    "welcome_message": "TEXT",
    "transcript_enabled": "INTEGER",
}


def _get_ticket_settings(guild_id: int) -> dict:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT category_id, log_channel_id, support_role_id, welcome_message, transcript_enabled "
        "FROM ticket_settings WHERE guild_id = ?",
        (guild_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return {
            "category_id": None,
            "log_channel_id": None,
            "support_role_id": None,
            "welcome_message": "Support will be with you shortly.",
            "transcript_enabled": "1",
        }
    category_id, log_channel_id, support_role_id, welcome_message, transcript_enabled = row
    return {
        "category_id": int(category_id) if category_id else None,
        "log_channel_id": int(log_channel_id) if log_channel_id else None,
        "support_role_id": int(support_role_id) if support_role_id else None,
        "welcome_message": welcome_message or "Support will be with you shortly.",
        "transcript_enabled": str(transcript_enabled) if transcript_enabled is not None else "1",
    }


def _set_ticket_settings(guild_id: int, **kwargs):
    allowed = set(TICKET_SETTINGS_COLUMNS.keys())
    updates = {k: v for k, v in kwargs.items() if k in allowed}
    if not updates:
        return
    columns = ", ".join(updates.keys())
    placeholders = ", ".join("?" for _ in updates)
    sql = (
        f"INSERT INTO ticket_settings (guild_id, {columns}) VALUES (?, {placeholders}) "
        f"ON CONFLICT(guild_id) DO UPDATE SET "
        + ", ".join(f"{col} = excluded.{col}" for col in updates.keys())
    )
    values = [guild_id] + list(updates.values())
    conn = get_db()
    conn.execute(sql, values)
    conn.commit()
    conn.close()


def _create_ticket(guild_id: int, channel_id: int, user_id: int):
    now = datetime.now(timezone.utc).timestamp()
    conn = get_db()
    conn.execute(
        "INSERT INTO tickets (guild_id, channel_id, user_id, created_at) VALUES (?, ?, ?, ?)",
        (guild_id, channel_id, user_id, now),
    )
    conn.commit()
    conn.close()


def _get_ticket_by_channel(channel_id: int) -> dict | None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT ticket_id, guild_id, channel_id, user_id, claimed_by, status, created_at FROM tickets WHERE channel_id = ? AND status = 'open'",
        (channel_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "ticket_id": row[0],
        "guild_id": row[1],
        "channel_id": row[2],
        "user_id": row[3],
        "claimed_by": row[4],
        "status": row[5],
        "created_at": row[6],
    }


def _get_ticket_by_user(guild_id: int, user_id: int) -> dict | None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT ticket_id, guild_id, channel_id, user_id, claimed_by, status, created_at FROM tickets WHERE guild_id = ? AND user_id = ? AND status = 'open'",
        (guild_id, user_id),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "ticket_id": row[0],
        "guild_id": row[1],
        "channel_id": row[2],
        "user_id": row[3],
        "claimed_by": row[4],
        "status": row[5],
        "created_at": row[6],
    }


def _claim_ticket(channel_id: int, user_id: int):
    conn = get_db()
    conn.execute(
        "UPDATE tickets SET claimed_by = ? WHERE channel_id = ?",
        (user_id, channel_id),
    )
    conn.commit()
    conn.close()


def _close_ticket_db(channel_id: int):
    now = datetime.now(timezone.utc).timestamp()
    conn = get_db()
    conn.execute(
        "UPDATE tickets SET status = 'closed', closed_at = ? WHERE channel_id = ?",
        (now, channel_id),
    )
    conn.commit()
    conn.close()


def _save_ticket_panel(guild_id: int, channel_id: int, message_id: int, label: str, emoji: str, description: str):
    conn = get_db()
    conn.execute(
        "INSERT INTO ticket_panels (guild_id, channel_id, message_id, label, emoji, description) VALUES (?, ?, ?, ?, ?, ?)",
        (guild_id, channel_id, message_id, label, emoji, description),
    )
    conn.commit()
    conn.close()


def _delete_ticket_db(channel_id: int):
    conn = get_db()
    conn.execute(
        "DELETE FROM tickets WHERE channel_id = ?",
        (channel_id,),
    )
    conn.commit()
    conn.close()


async def _generate_transcript(channel: discord.TextChannel, limit: int = 100) -> str:
    lines = []
    async for msg in channel.history(limit=limit, oldest_first=True):
        timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M:%S")
        content = msg.content or ""
        lines.append(f"[{timestamp}] {msg.author.display_name}: {content}")
        for att in msg.attachments:
            lines.append(f"[{timestamp}] {msg.author.display_name} attached: {att.url}")
    return "\n".join(lines)


def _ticket_overwrites(guild: discord.Guild, user: discord.Member, support_role_id: int | None):
    overwrites = {
        guild.default_role: discord.PermissionOverwrite(view_channel=False),
        user: discord.PermissionOverwrite(view_channel=True, send_messages=True, read_message_history=True),
        guild.me: discord.PermissionOverwrite(view_channel=True, send_messages=True, manage_channels=True, manage_messages=True),
    }
    if support_role_id:
        role = guild.get_role(int(support_role_id))
        if role:
            overwrites[role] = discord.PermissionOverwrite(view_channel=True, send_messages=True, read_message_history=True)
    return overwrites


async def _create_ticket_channel(guild: discord.Guild, user: discord.Member, settings: dict) -> discord.TextChannel:
    category_id = settings.get("category_id")
    support_role_id = settings.get("support_role_id")
    category = guild.get_channel(int(category_id)) if category_id else None
    overwrites = _ticket_overwrites(guild, user, support_role_id)
    safe_name = re.sub(r"[^a-z0-9\-]", "", user.display_name.lower().replace(" ", "-"))[:40]
    if not safe_name:
        safe_name = "user"
    channel_name = f"ticket-{safe_name}"
    suffix = 1
    while discord.utils.get(guild.text_channels, name=channel_name):
        channel_name = f"ticket-{safe_name}-{suffix}"
        suffix += 1
    if category and isinstance(category, discord.CategoryChannel):
        channel = await guild.create_text_channel(channel_name, category=category, overwrites=overwrites)
    else:
        channel = await guild.create_text_channel(channel_name, overwrites=overwrites)
    return channel


async def _send_ticket_welcome(channel: discord.TextChannel, user: discord.Member, settings: dict):
    welcome = settings.get("welcome_message") or "Support will be with you shortly."
    embed = discord.Embed(
        title=":ticket: Ticket Opened",
        description=f"Hey {user.mention},\n{welcome}",
        color=discord.Color.green(),
    )
    view = TicketControlView()
    msg = await channel.send(embed=embed, view=view)
    await msg.pin()


class TicketControlView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)

    @discord.ui.button(label="Claim", style=discord.ButtonStyle.primary, emoji="👤", custom_id="ticket_claim")
    async def claim(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        ticket = await asyncio.to_thread(_get_ticket_by_channel, interaction.channel_id)
        if not ticket:
            try:
                await interaction.followup.send(":x: This is not a ticket channel.", ephemeral=True)
            except Exception:
                pass
            return
        if ticket["claimed_by"]:
            claimer = interaction.guild.get_member(int(ticket["claimed_by"]))
            name = claimer.mention if claimer else "someone"
            await interaction.followup.send(f":x: Already claimed by {name}.", ephemeral=True)
            return
        await asyncio.to_thread(_claim_ticket, interaction.channel_id, interaction.user.id)
        await interaction.followup.send(f":white_check_mark: Claimed by {interaction.user.mention}")

    @discord.ui.button(label="Close", style=discord.ButtonStyle.danger, emoji="🔒", custom_id="ticket_close")
    async def close(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        ticket = await asyncio.to_thread(_get_ticket_by_channel, interaction.channel_id)
        if not ticket:
            try:
                await interaction.followup.send(":x: This is not a ticket channel.", ephemeral=True)
            except Exception:
                pass
            return
        view = ConfirmCloseView()
        await interaction.followup.send("Are you sure you want to close this ticket?", view=view, ephemeral=True)


class ConfirmCloseView(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=60)

    @discord.ui.button(label="Yes, close", style=discord.ButtonStyle.danger, custom_id="ticket_confirm_close")
    async def confirm(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        channel = interaction.channel
        ticket = await asyncio.to_thread(_get_ticket_by_channel, channel.id)
        if not ticket:
            try:
                await interaction.followup.send(":x: Ticket not found.", ephemeral=True)
            except Exception:
                pass
            return
        settings = await asyncio.to_thread(_get_ticket_settings, interaction.guild.id)
        transcript = ""
        if settings.get("transcript_enabled") == "1":
            transcript = await _generate_transcript(channel)
        await asyncio.to_thread(_close_ticket_db, channel.id)
        log_channel_id = settings.get("log_channel_id")
        if log_channel_id:
            log_ch = interaction.guild.get_channel(int(log_channel_id))
            if log_ch:
                user = interaction.guild.get_member(int(ticket["user_id"]))
                user_name = user.mention if user else f"User {ticket['user_id']}"
                embed = discord.Embed(
                    title=":ticket: Ticket Closed",
                    description=f"Ticket {channel.name} closed by {interaction.user.mention}",
                    color=discord.Color.red(),
                )
                embed.add_field(name="Opened by", value=user_name, inline=True)
                if ticket["claimed_by"]:
                    claimer = interaction.guild.get_member(int(ticket["claimed_by"]))
                    embed.add_field(name="Claimed by", value=claimer.mention if claimer else "Unknown", inline=True)
                if transcript:
                    buffer = io.BytesIO(transcript.encode("utf-8"))
                    await log_ch.send(embed=embed, file=discord.File(buffer, filename=f"{channel.name}-transcript.txt"))
                else:
                    await log_ch.send(embed=embed)
        await interaction.followup.send(":white_check_mark: Ticket closed.")
        await channel.delete()

    @discord.ui.button(label="Cancel", style=discord.ButtonStyle.secondary, custom_id="ticket_cancel_close")
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.send_message(":x: Close cancelled.", ephemeral=True)


class TicketPanelView(discord.ui.View):
    def __init__(self, panel_id: int):
        super().__init__(timeout=None)
        self.panel_id = panel_id

    @discord.ui.button(label="Create Ticket", style=discord.ButtonStyle.green, emoji="🎫", custom_id="ticket_panel_create")
    async def create_ticket(self, interaction: discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(ephemeral=True)
        settings = await asyncio.to_thread(_get_ticket_settings, interaction.guild.id)
        if not settings.get("category_id"):
            await interaction.followup.send(":x: Ticket system is not set up. Ask an admin to run `/ticketsetup`.", ephemeral=True)
            return
        existing = await asyncio.to_thread(_get_ticket_by_user, interaction.guild.id, interaction.user.id)
        if existing:
            channel = interaction.guild.get_channel(int(existing["channel_id"]))
            if channel:
                await interaction.followup.send(f":x: You already have an open ticket: {channel.mention}", ephemeral=True)
                return
        channel = await _create_ticket_channel(interaction.guild, interaction.user, settings)
        await asyncio.gather(
            asyncio.to_thread(_create_ticket, interaction.guild.id, channel.id, interaction.user.id),
            _send_ticket_welcome(channel, interaction.user, settings),
        )
        await interaction.followup.send(f":white_check_mark: Ticket created: {channel.mention}", ephemeral=True)


# --- Admin action execution from chat ---

ADMIN_ACTION_HELPERS: dict[str, callable] = {}


def _parse_duration(duration_str: str) -> timedelta | None:
    duration_str = duration_str.strip().lower()
    units = {"s": 1, "sec": 1, "m": 60, "min": 60, "h": 3600, "hr": 3600, "d": 86400, "day": 86400}
    total_seconds = 0
    parts = re.findall(r"(\d+)\s*([a-z]+)", duration_str)
    if not parts:
        try:
            total_seconds = int(duration_str) * 3600
        except ValueError:
            return None
    else:
        for value, unit in parts:
            multiplier = units.get(unit.rstrip("s"), 0)
            if not multiplier:
                return None
            total_seconds += int(value) * multiplier
    if total_seconds <= 0:
        return None
    return timedelta(seconds=total_seconds)


def _is_guild_admin(author: discord.Member) -> bool:
    return author == author.guild.owner or author.guild_permissions.administrator


def _can_moderate_member(author: discord.Member, target: discord.Member) -> tuple[bool, str]:
    """Check if author can moderate target. Returns (ok, error_message)."""
    if _is_guild_admin(author):
        return True, ""
    if author.top_role <= target.top_role:
        return False, ":x: You can't moderate someone with a higher or equal role."
    return True, ""


async def _action_kick(message: discord.Message, target: discord.Member, reason: str) -> str:
    if not message.author.guild_permissions.kick_members and not _is_guild_admin(message.author):
        return ":x: You don't have permission to kick members."
    if not message.guild.me.guild_permissions.kick_members:
        return ":x: I don't have permission to kick members."
    ok, err = _can_moderate_member(message.author, target)
    if not ok:
        return err
    try:
        await target.kick(reason=f"{message.author}: {reason}"[:512])
        return f":white_check_mark: Kicked {target.mention}."
    except Exception as e:
        return f":x: Failed to kick: {e}"


async def _action_ban(message: discord.Message, target: discord.Member, reason: str) -> str:
    if not message.author.guild_permissions.ban_members and not _is_guild_admin(message.author):
        return ":x: You don't have permission to ban members."
    if not message.guild.me.guild_permissions.ban_members:
        return ":x: I don't have permission to ban members."
    ok, err = _can_moderate_member(message.author, target)
    if not ok:
        return err
    try:
        await target.ban(reason=f"{message.author}: {reason}"[:512])
        return f":white_check_mark: Banned {target.mention}."
    except Exception as e:
        return f":x: Failed to ban: {e}"


async def _action_timeout(message: discord.Message, target: discord.Member, duration_str: str, reason: str) -> str:
    if not message.author.guild_permissions.moderate_members and not _is_guild_admin(message.author):
        return ":x: You don't have permission to timeout members."
    if not message.guild.me.guild_permissions.moderate_members:
        return ":x: I don't have permission to timeout members."
    duration = _parse_duration(duration_str)
    if not duration:
        return ":x: Invalid duration. Use like `1h`, `30m`, `1d`."
    if duration > timedelta(days=28):
        return ":x: Timeout can't be longer than 28 days."
    ok, err = _can_moderate_member(message.author, target)
    if not ok:
        return err
    try:
        until = datetime.now(timezone.utc) + duration
        await target.timeout(until, reason=f"{message.author}: {reason}"[:512])
        return f":white_check_mark: Timed out {target.mention} for {duration_str}."
    except Exception as e:
        return f":x: Failed to timeout: {e}"


async def _action_mute(message: discord.Message, target: discord.Member, reason: str) -> str:
    """Text-chat mute is implemented as a 1-hour timeout by default."""
    if not message.author.guild_permissions.moderate_members and not _is_guild_admin(message.author):
        return ":x: You don't have permission to mute members."
    if not message.guild.me.guild_permissions.moderate_members:
        return ":x: I don't have permission to mute members."
    ok, err = _can_moderate_member(message.author, target)
    if not ok:
        return err
    try:
        until = datetime.now(timezone.utc) + timedelta(hours=1)
        await target.timeout(until, reason=f"{message.author}: {reason}"[:512])
        return f":white_check_mark: Muted {target.mention} for 1 hour."
    except Exception as e:
        return f":x: Failed to mute: {e}"


async def _action_voicemute(message: discord.Message, target: discord.Member, reason: str) -> str:
    if not message.author.guild_permissions.mute_members and not _is_guild_admin(message.author):
        return ":x: You don't have permission to server-mute members."
    if not message.guild.me.guild_permissions.mute_members:
        return ":x: I don't have permission to server-mute members."
    ok, err = _can_moderate_member(message.author, target)
    if not ok:
        return err
    try:
        await target.edit(mute=True, reason=f"{message.author}: {reason}"[:512])
        return f":white_check_mark: Server-muted {target.mention}."
    except Exception as e:
        return f":x: Failed to server-mute: {e}"


async def _action_voiceunmute(message: discord.Message, target: discord.Member, reason: str) -> str:
    if not message.author.guild_permissions.mute_members and not _is_guild_admin(message.author):
        return ":x: You don't have permission to unmute members."
    if not message.guild.me.guild_permissions.mute_members:
        return ":x: I don't have permission to unmute members."
    ok, err = _can_moderate_member(message.author, target)
    if not ok:
        return err
    try:
        await target.edit(mute=False, reason=f"{message.author}: {reason}"[:512])
        return f":white_check_mark: Unmuted {target.mention}."
    except Exception as e:
        return f":x: Failed to unmute: {e}"


async def _action_deafen(message: discord.Message, target: discord.Member, reason: str) -> str:
    if not message.author.guild_permissions.deafen_members and not _is_guild_admin(message.author):
        return ":x: You don't have permission to server-deafen members."
    if not message.guild.me.guild_permissions.deafen_members:
        return ":x: I don't have permission to server-deafen members."
    ok, err = _can_moderate_member(message.author, target)
    if not ok:
        return err
    try:
        await target.edit(deafen=True, reason=f"{message.author}: {reason}"[:512])
        return f":white_check_mark: Server-deafened {target.mention}."
    except Exception as e:
        return f":x: Failed to server-deafen: {e}"


async def _action_undeafen(message: discord.Message, target: discord.Member, reason: str) -> str:
    if not message.author.guild_permissions.deafen_members and not _is_guild_admin(message.author):
        return ":x: You don't have permission to undeafen members."
    if not message.guild.me.guild_permissions.deafen_members:
        return ":x: I don't have permission to undeafen members."
    ok, err = _can_moderate_member(message.author, target)
    if not ok:
        return err
    try:
        await target.edit(deafen=False, reason=f"{message.author}: {reason}"[:512])
        return f":white_check_mark: Undeafened {target.mention}."
    except Exception as e:
        return f":x: Failed to undeafen: {e}"


async def _action_unban(message: discord.Message, user_id_or_mention: str) -> str:
    if not message.author.guild_permissions.ban_members:
        return ":x: You don't have permission to unban members."
    if not message.guild.me.guild_permissions.ban_members:
        return ":x: I don't have permission to unban members."
    user_id = _extract_user_id(user_id_or_mention)
    if not user_id:
        return ":x: Could not find user. Provide an ID or mention."
    try:
        ban_entry = await message.guild.fetch_ban(discord.Object(id=user_id))
        await message.guild.unban(ban_entry.user, reason=f"{message.author}")
        return f":white_check_mark: Unbanned {ban_entry.user.mention}."
    except Exception as e:
        return f":x: Failed to unban: {e}"


async def _action_create_channel(message: discord.Message, name: str, channel_type: str) -> str:
    if not message.author.guild_permissions.manage_channels:
        return ":x: You don't have permission to manage channels."
    if not message.guild.me.guild_permissions.manage_channels:
        return ":x: I don't have permission to manage channels."
    name = name.strip("#")
    try:
        if channel_type.lower() in ("voice", "vc"):
            ch = await message.guild.create_voice_channel(name)
        elif channel_type.lower() in ("category", "cat"):
            ch = await message.guild.create_category(name)
        else:
            ch = await message.guild.create_text_channel(name)
        return f":white_check_mark: Created {ch.mention}."
    except Exception as e:
        return f":x: Failed to create channel: {e}"


async def _action_move_channel(message: discord.Message, channel: discord.abc.GuildChannel, target: discord.abc.GuildChannel, direction: str) -> str:
    if not message.author.guild_permissions.manage_channels:
        return ":x: You don't have permission to manage channels."
    if not message.guild.me.guild_permissions.manage_channels:
        return ":x: I don't have permission to manage channels."
    try:
        if direction == "before":
            await channel.move(before=target)
            return f":white_check_mark: Moved {channel.mention} above {target.mention}."
        elif direction == "after":
            await channel.move(after=target)
            return f":white_check_mark: Moved {channel.mention} below {target.mention}."
        elif direction == "to":
            await channel.move(beginning=True)
            return f":white_check_mark: Moved {channel.mention} to top."
        else:
            await channel.move(before=target)
            return f":white_check_mark: Moved {channel.mention} before {target.mention}."
    except Exception as e:
        return f":x: Failed to move channel: {e}"


async def _action_delete_channel(message: discord.Message, channel: discord.TextChannel | discord.VoiceChannel | discord.CategoryChannel) -> str:
    if not message.author.guild_permissions.manage_channels:
        return ":x: You don't have permission to manage channels."
    if not message.guild.me.guild_permissions.manage_channels:
        return ":x: I don't have permission to manage channels."
    try:
        await channel.delete(reason=f"{message.author}")
        return f":white_check_mark: Deleted #{channel.name}."
    except Exception as e:
        return f":x: Failed to delete channel: {e}"


async def _action_lock(message: discord.Message, channel: discord.TextChannel) -> str:
    if not message.author.guild_permissions.manage_channels:
        return ":x: You don't have permission to manage channels."
    if not message.guild.me.guild_permissions.manage_channels:
        return ":x: I don't have permission to manage channels."
    try:
        await channel.set_permissions(channel.guild.default_role, send_messages=False)
        return f":white_check_mark: Locked {channel.mention}."
    except Exception as e:
        return f":x: Failed to lock: {e}"


async def _action_unlock(message: discord.Message, channel: discord.TextChannel) -> str:
    if not message.author.guild_permissions.manage_channels:
        return ":x: You don't have permission to manage channels."
    if not message.guild.me.guild_permissions.manage_channels:
        return ":x: I don't have permission to manage channels."
    try:
        await channel.set_permissions(channel.guild.default_role, send_messages=None)
        return f":white_check_mark: Unlocked {channel.mention}."
    except Exception as e:
        return f":x: Failed to unlock: {e}"


async def _action_slowmode(message: discord.Message, channel: discord.TextChannel, seconds: str) -> str:
    if not message.author.guild_permissions.manage_channels:
        return ":x: You don't have permission to manage channels."
    if not message.guild.me.guild_permissions.manage_channels:
        return ":x: I don't have permission to manage channels."
    try:
        secs = int(seconds)
        await channel.edit(slowmode_delay=secs)
        return f":white_check_mark: Set slowmode in {channel.mention} to {secs} seconds."
    except Exception as e:
        return f":x: Failed to set slowmode: {e}"


async def _action_purge(message: discord.Message, amount: str) -> str:
    if not message.author.guild_permissions.manage_messages:
        return ":x: You don't have permission to manage messages."
    if not message.guild.me.guild_permissions.manage_messages:
        return ":x: I don't have permission to manage messages."
    try:
        count = int(amount)
        if count < 1 or count > 100:
            return ":x: Purge amount must be between 1 and 100."
        deleted = await message.channel.purge(limit=count + 1)
        return f":white_check_mark: Deleted {len(deleted)} messages."
    except Exception as e:
        return f":x: Failed to purge: {e}"


async def _action_addrole(message: discord.Message, target: discord.Member, role: discord.Role) -> str:
    if not message.author.guild_permissions.manage_roles and not _is_guild_admin(message.author):
        return ":x: You don't have permission to manage roles."
    if not message.guild.me.guild_permissions.manage_roles:
        return ":x: I don't have permission to manage roles."
    if not _is_guild_admin(message.author) and message.author.top_role <= role:
        return ":x: You can't assign a role higher than or equal to your top role."
    if message.guild.me.top_role <= role:
        return ":x: My role is too low to assign that role."
    try:
        await target.add_roles(role, reason=f"{message.author}")
        return f":white_check_mark: Added {role.mention} to {target.mention}."
    except Exception as e:
        return f":x: Failed to add role: {e}"


async def _action_removerole(message: discord.Message, target: discord.Member, role: discord.Role) -> str:
    if not message.author.guild_permissions.manage_roles and not _is_guild_admin(message.author):
        return ":x: You don't have permission to manage roles."
    if not message.guild.me.guild_permissions.manage_roles:
        return ":x: I don't have permission to manage roles."
    if not _is_guild_admin(message.author) and message.author.top_role <= role:
        return ":x: You can't remove a role higher than or equal to your top role."
    if message.guild.me.top_role <= role:
        return ":x: My role is too low to remove that role."
    try:
        await target.remove_roles(role, reason=f"{message.author}")
        return f":white_check_mark: Removed {role.mention} from {target.mention}."
    except Exception as e:
        return f":x: Failed to remove role: {e}"


def _extract_user_id(text: str) -> int | None:
    text = text.strip()
    match = re.match(r"<@!?(\d+)>", text)
    if match:
        return int(match.group(1))
    if text.isdigit():
        return int(text)
    return None


def _resolve_member(guild: discord.Guild, text: str, author: discord.Member | None = None) -> discord.Member | None:
    text = text.strip()
    if text.lower() == "me" and author is not None:
        return author
    user_id = _extract_user_id(text)
    if user_id:
        return guild.get_member(user_id)
    text_lower = text.lower().strip("@!#")
    for member in guild.members:
        if member.name.lower() == text_lower or member.display_name.lower() == text_lower:
            return member
    # Partial match fallback
    for member in guild.members:
        if text_lower in member.name.lower() or text_lower in member.display_name.lower():
            return member
    return None


def _resolve_role(guild: discord.Guild, text: str) -> discord.Role | None:
    text = text.strip()
    match = re.match(r"<@&(\d+)>", text)
    if match:
        return guild.get_role(int(match.group(1)))
    text_lower = text.lower().strip("@")
    for role in guild.roles:
        if role.name.lower() == text_lower:
            return role
    return None


def _resolve_channel(guild: discord.Guild, text: str) -> discord.abc.GuildChannel | None:
    text = text.strip()
    match = re.match(r"<#(\d+)>", text)
    if match:
        return guild.get_channel(int(match.group(1)))
    text_lower = text.lower().strip("#")
    for channel in guild.channels:
        if channel.name.lower() == text_lower:
            return channel
    return None


def _extract_action_lines(text: str) -> list[tuple[str, list[str]]]:
    actions = []
    for line in text.split("\n"):
        line = line.strip()
        if not line.startswith("ACTION:"):
            continue
        action_text = line[7:].strip()
        parts = action_text.split(None, 2)
        if not parts:
            continue
        actions.append((parts[0].lower(), parts[1:]))
    return actions


def _guess_admin_actions(content: str) -> list[tuple[str, list[str]]]:
    """Fallback pattern matcher for direct admin requests when the AI doesn't produce ACTION lines."""
    guessed = []
    stop_words = {"in", "and", "or", "from", "to", "the", "a", "an", "on", "at", "for", "with"}

    def _clean_target(raw: str) -> str | None:
        raw = raw.strip().rstrip(",.!?")
        if raw.lower() == "me":
            return "me"
        if raw.lower() in stop_words:
            return None
        return raw

    # Voice mute: mute ... voice / vc
    if re.search(r"\bmute\b", content, re.IGNORECASE) and re.search(r"\b(voice|vc)\b", content, re.IGNORECASE):
        for m in re.finditer(r"<@!?\d+>", content):
            target = _clean_target(m.group(0))
            if target:
                guessed.append(("voicemute", [target]))
        # Also try names after "user"
        for m in re.finditer(r"\buser\s+(.+?)(?:\s+(?:in|and|or|from|to)\b|$)", content, re.IGNORECASE):
            target = _clean_target(m.group(1).split()[0])
            if target:
                guessed.append(("voicemute", [target]))

    # Deafen: deafen ...
    if re.search(r"\bdeafen\b", content, re.IGNORECASE):
        for m in re.finditer(r"<@!?\d+>", content):
            target = _clean_target(m.group(0))
            if target:
                guessed.append(("deafen", [target]))
        for m in re.finditer(r"\bdeafen\s+(?:the\s+)?(?:user\s+)?(.+?)(?:\s+(?:in|and|or|from|to)\b|$)", content, re.IGNORECASE):
            target = _clean_target(m.group(1).split()[0])
            if target:
                guessed.append(("deafen", [target]))

    # Plain text mute (no voice mentioned)
    if re.search(r"\bmute\b", content, re.IGNORECASE) and not re.search(r"\b(voice|vc)\b", content, re.IGNORECASE):
        for m in re.finditer(r"<@!?\d+>", content):
            target = _clean_target(m.group(0))
            if target:
                guessed.append(("mute", [target]))

    return guessed


async def execute_admin_actions(message: discord.Message, text: str) -> list[str]:
    if not message.guild:
        return []
    results = []
    action_lines = _extract_action_lines(text)
    if not action_lines:
        action_lines = _guess_admin_actions(message.content)
    for action, args in action_lines:
        result = await _dispatch_admin_action(message, action, args)
        if result:
            results.append(result)
    return results


async def _dispatch_admin_action(message: discord.Message, action: str, args: list[str]) -> str:
    guild = message.guild
    try:
        if action == "kick" and len(args) >= 1:
            target = _resolve_member(guild, args[0], message.author)
            reason = args[1] if len(args) > 1 else "No reason provided"
            if not target:
                return ":x: Member not found."
            return await _action_kick(message, target, reason)

        if action == "ban" and len(args) >= 1:
            target = _resolve_member(guild, args[0], message.author)
            reason = args[1] if len(args) > 1 else "No reason provided"
            if not target:
                return ":x: Member not found."
            return await _action_ban(message, target, reason)

        if action == "timeout" and len(args) >= 2:
            target = _resolve_member(guild, args[0], message.author)
            duration = args[1]
            reason = args[2] if len(args) > 2 else "No reason provided"
            if not target:
                return ":x: Member not found."
            return await _action_timeout(message, target, duration, reason)

        if action == "mute" and len(args) >= 1:
            target = _resolve_member(guild, args[0], message.author)
            reason = args[1] if len(args) > 1 else "No reason provided"
            if not target:
                return ":x: Member not found."
            return await _action_mute(message, target, reason)

        if action == "voicemute" and len(args) >= 1:
            target = _resolve_member(guild, args[0], message.author)
            reason = args[1] if len(args) > 1 else "No reason provided"
            if not target:
                return ":x: Member not found."
            return await _action_voicemute(message, target, reason)

        if action == "voiceunmute" and len(args) >= 1:
            target = _resolve_member(guild, args[0], message.author)
            reason = args[1] if len(args) > 1 else "No reason provided"
            if not target:
                return ":x: Member not found."
            return await _action_voiceunmute(message, target, reason)

        if action == "deafen" and len(args) >= 1:
            target = _resolve_member(guild, args[0], message.author)
            reason = args[1] if len(args) > 1 else "No reason provided"
            if not target:
                return ":x: Member not found."
            return await _action_deafen(message, target, reason)

        if action == "undeafen" and len(args) >= 1:
            target = _resolve_member(guild, args[0], message.author)
            reason = args[1] if len(args) > 1 else "No reason provided"
            if not target:
                return ":x: Member not found."
            return await _action_undeafen(message, target, reason)

        if action == "unban" and len(args) >= 1:
            return await _action_unban(message, args[0])

        if action == "remember" and len(args) >= 3 and args[0].lower() == "note":
            note_key = args[1].rstrip(":")
            note_value = " ".join(args[2:])
            _save_user_note(message.author.id, note_key, note_value)
            return f":white_check_mark: Remembered: {note_key} = {note_value}"

        if action == "create" and len(args) >= 2 and args[0].lower() == "channel":
            name = args[1]
            if len(args) > 2:
                channel_type = args[2]
            elif any(w in name.lower() for w in ["voice", "vc"]):
                channel_type = "voice"
            else:
                channel_type = "text"
            return await _action_create_channel(message, name, channel_type)

        if action == "move" and len(args) >= 3:
            channel = _resolve_channel(guild, args[0])
            direction = args[1].lower()
            target = _resolve_channel(guild, args[2])
            if not channel:
                return ":x: Channel not found."
            if not target and direction in ("above", "below"):
                return ":x: Target channel not found."
            return await _action_move_channel(message, channel, target or channel, direction)

        if action == "delete" and len(args) >= 2 and args[0].lower() == "channel":
            channel = _resolve_channel(guild, args[1])
            if not channel:
                return ":x: Channel not found."
            return await _action_delete_channel(message, channel)

        if action == "lock" and len(args) >= 1:
            channel = _resolve_channel(guild, args[0])
            if not isinstance(channel, discord.TextChannel):
                return ":x: Text channel not found."
            return await _action_lock(message, channel)

        if action == "unlock" and len(args) >= 1:
            channel = _resolve_channel(guild, args[0])
            if not isinstance(channel, discord.TextChannel):
                return ":x: Text channel not found."
            return await _action_unlock(message, channel)

        if action == "slowmode" and len(args) >= 2:
            channel = _resolve_channel(guild, args[0])
            if not isinstance(channel, discord.TextChannel):
                return ":x: Text channel not found."
            return await _action_slowmode(message, channel, args[1])

        if action == "purge" and len(args) >= 1:
            return await _action_purge(message, args[0])

        if action == "addrole" and len(args) >= 2:
            target = _resolve_member(guild, args[0], message.author)
            role = _resolve_role(guild, args[1])
            if not target or not role:
                return ":x: Member or role not found."
            return await _action_addrole(message, target, role)

        if action == "removerole" and len(args) >= 2:
            target = _resolve_member(guild, args[0], message.author)
            role = _resolve_role(guild, args[1])
            if not target or not role:
                return ":x: Member or role not found."
            return await _action_removerole(message, target, role)

        if action == "joinvoice":
            if not HAS_VOICE_RECV:
                return ":x: voice_recv library not installed."
            if not message.author.voice:
                return ":x: You are not in a voice channel."
            try:
                guild = message.guild
                existing = guild.voice_client
                if existing:
                    try: existing.stop()
                    except: pass
                    try: await existing.disconnect()
                    except: pass
                voice = message.author.voice.channel
                try:
                    vc = await voice.connect(cls=_voice_recv.VoiceRecvClient)
                except Exception as e1:
                    err_str = str(e1).lower()
                    if "wavelink" in err_str or "node" in err_str or "pool" in err_str:
                        import wavelink as _wl
                        import asyncio as _asyncio
                        _fallback_nodes = [
                            _wl.Node(uri="http://lavalink.jirayu.net:13592", password="youshallnotpass", retries=1),
                            _wl.Node(uri="http://lavalinkv4.serenetia.com:80", password="https://seretia.link/discord", retries=1),
                        ]
                        for _n in _fallback_nodes:
                            try:
                                await _asyncio.wait_for(_wl.Pool.connect(nodes=[_n], client=bot), timeout=15)
                            except:
                                continue
                        vc = await voice.connect(cls=_voice_recv.VoiceRecvClient)
                    else:
                        return f":x: Could not join voice: {e1}"
                _voice_text_channels[guild.id] = message.channel.id
                _voice_tts_lang.setdefault(guild.id, "he")
                loop = asyncio.get_running_loop()
                guild_id = guild.id
                lock = _voice_recv_locks.setdefault(guild.id, asyncio.Lock())
                def make_text_cb():
                    async def text_cb_async(user, text):
                        if not text or len(text.strip()) < 2:
                            return
                        async with _voice_recv_locks.setdefault(guild_id, asyncio.Lock()):
                            if not _voice_recv_active.get(guild_id):
                                return
                            await _handle_voice_speech(guild_id, user, text.strip(), loop)
                    def text_cb(user, text):
                        try:
                            coro = text_cb_async(user, text)
                            asyncio.run_coroutine_threadsafe(coro, loop)
                        except Exception:
                            pass
                    return text_cb
                sink = SpeechRecognitionSink(text_cb=make_text_cb(), process_cb=_make_process_cb(), default_recognizer='google', phrase_time_limit=15)
                vc.listen(sink)
                _voice_recv_active[guild.id] = True
                return f":loud_sound: Joined **{voice}** and listening."
            except Exception as e:
                return f":x: Could not join voice: {e}"

        return ""
    except Exception as e:
        return f":x: Admin action error: {e}"


async def fetch_avatar(url: str) -> bytes:
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.read()


async def read_attachment(att: discord.Attachment, max_bytes: int = 50000) -> tuple[str, str] | None:
    try:
        if att.size > max_bytes:
            return None
        binary_exts = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".mp3", ".mp4", ".wav", ".avi", ".mkv", ".pdf", ".exe", ".dll", ".zip", ".rar", ".7z"}
        if any(str(att.filename).lower().endswith(e) for e in binary_exts):
            return None
        async with aiohttp.ClientSession() as session:
            async with session.get(att.url) as resp:
                data = await resp.read()
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = data.decode("utf-8", errors="replace")
            except Exception:
                return None
        return att.filename, text[:4000]
    except Exception as e:
        print(f"Failed to read attachment {att.filename}: {e}")
        return None


def _get_font(size: int):
    for font_name in ["arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            pass
    return ImageFont.load_default()


async def generate_rank_card(member: discord.Member, data: dict) -> io.BytesIO:
    width, height = 800, 250
    bg = Image.new("RGB", (width, height), "#1a1b26")
    draw = ImageDraw.Draw(bg)

    # Accent bar
    draw.rectangle([0, 0, 12, height], fill="#7aa2f7")

    # Avatar
    avatar_data = await fetch_avatar(str(member.display_avatar.url))
    avatar = Image.open(io.BytesIO(avatar_data)).convert("RGBA")
    avatar = avatar.resize((140, 140))
    mask = Image.new("L", (140, 140), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.ellipse([0, 0, 140, 140], fill=255)
    bg.paste(avatar, (40, 55), mask)

    # Border circle
    draw.ellipse([38, 53, 182, 197], outline="#7aa2f7", width=4)

    font_big = _get_font(36)
    font_mid = _get_font(26)
    font_small = _get_font(20)

    # Name and rank
    draw.text((210, 40), member.display_name, fill="#c0caf5", font=font_big)
    draw.text((210, 85), f"Rank #{data['rank']}  •  Level {data['level']}", fill="#a9b1d6", font=font_mid)

    # XP info
    current_xp = data["xp"]
    current_level = data["level"]
    prev_xp = xp_for_level(current_level)
    next_xp = xp_for_level(current_level + 1)
    xp_in_level = current_xp - prev_xp
    xp_needed = next_xp - prev_xp
    progress = min(1.0, max(0.0, xp_in_level / xp_needed)) if xp_needed > 0 else 1.0

    bar_y = 160
    bar_width = 540
    bar_height = 24
    draw.rounded_rectangle([210, bar_y, 210 + bar_width, bar_y + bar_height], radius=12, fill="#24283b")
    fill_width = int(bar_width * progress)
    draw.rounded_rectangle([210, bar_y, 210 + fill_width, bar_y + bar_height], radius=12, fill="#7aa2f7")

    xp_text = f"{current_xp} / {next_xp} XP"
    draw.text((210, bar_y + 32), xp_text, fill="#a9b1d6", font=font_small)
    draw.text((210 + bar_width - 120, bar_y + 32), f"{int(progress * 100)}%", fill="#a9b1d6", font=font_small)

    # Stats
    draw.text((650, 50), f"Msgs: {data['messages']}", fill="#a9b1d6", font=font_small)
    draw.text((650, 78), f"Voice: {data['voice_minutes']}m", fill="#a9b1d6", font=font_small)

    buf = io.BytesIO()
    bg.save(buf, format="PNG")
    buf.seek(0)
    return buf


def get_history(channel_id: int) -> list[dict]:
    if channel_id not in message_history:
        return []
    return list(message_history[channel_id])


def add_to_history(channel_id: int, role: str, content: str):
    if channel_id not in message_history:
        message_history[channel_id] = deque(maxlen=MAX_HISTORY)
    message_history[channel_id].append({"role": role, "content": content})

CHAT_CAPABILITIES = """
WHAT I CAN DO:
- /play, /skip, /stop, /queue, /pause, /resume, /volume — music from YouTube
- /search or just ask — I'll search the web if I need current info
- /rank, /leaderboard — XP for chatting and voice, role rewards, multipliers
- /lua, /script, /file — generate any code or file (just say "make me a .py file...")
- /read #channel — I'll summarize recent messages
- /voice — see who's in vc right now
- Admin stuff: kick/ban/timeout/mute/voicemute/deafen/create/delete/lock/unlock/slowmode/addrole/removerole/purge — just ask in chat
- /ticket — ticket system with claims and transcripts
- /valorantsearch, /valorant, /valorantmmr, /valorantmatches — Valorant stats
- I see images you attach, I read text files you drop
- I can generate images from descriptions (try "generate an image of...")
- I remember the last 50 messages in each channel

I'm a real bot with real features. If I say "I can't" and you know I can, call me out.
"""

CHAT_SYSTEM = textwrap.dedent(f"""\
You're Null — a Discord bot that's actually cool to talk to. Quick, witty, helpful. No robotic vibes.
{CHAT_CAPABILITIES}

LANGUAGE RULES (IMPORTANT):
- ALWAYS respond in the EXACT same language the user wrote to you in.
- If they write in Hebrew (א-ת), respond in Hebrew. NEVER respond in Arabic (ا-ي) or any other language.
- If they write in Arabic (ا-ي), respond in Arabic. NEVER respond in Hebrew.
- If they write in English, respond in English.
- Voice messages can also be in Hebrew or English — match the spoken language.
- This bot's default voice TTS language is Hebrew (he-IL-HilaNeural), so default to Hebrew in voice conversations unless the user speaks English.
- These two languages look similar but are completely different. Pay close attention to the script.
- When in doubt, match the user's exact language character by character.

CODING & FILES:
- For code/file requests ("make me a...", "write a script..."), output the code. The system will send it as a file automatically.
- If someone asks "what is this" or "what does this do" about code you just sent, explain it in text. Don't generate new code.
- Don't output code blocks unless the user asked for code.

ADMIN ACTIONS (MUST follow this exactly):
- When the user tells you to kick/ban/timeout/mute/voicemute/deafen/undeafen/create/delete/lock/unlock/slowmode/addrole/removerole/purge someone/something, put an ACTION: line at the end of your reply.
- Format: ACTION: kick @user reason | ACTION: timeout @user 1h reason | ACTION: mute @user reason | ACTION: voicemute @user reason | ACTION: deafen @user reason | ACTION: create channel #name text | ACTION: lock #channel | ACTION: slowmode #channel 5 | ACTION: purge 10 | etc.
- "mute" without "voice" = ACTION: mute @user (text timeout, 1h)
- "mute" with "voice"/"vc" = ACTION: voicemute @user
- NEVER run an action without an ACTION: line. The user has to ask first.

FACTS & INFO:
- If it's current events, news, sports, weather, dates — use the web search results I give you. Don't guess from old training data.
- If I gave you search results, trust them over your built-in knowledge.
- If no search results and it's time-sensitive, just say you don't have current info.
- If I gave you voice channel data, use it to answer VC questions. Don't say "I can't see that."
- For Valorant stats, direct them to /valorant or /valorantmmr with name+tag.
- Keep it short and natural. Don't list all features unless they ask.
- Be yourself. You're not a robot reading a script.

CRITICAL RULES:
- When the user asks you to "read this", "copy this", "say this", or "read out loud" — just DO IT. Output the text directly. Don't summarize it, don't add safety warnings, don't analyze it, don't lecture.
- NEVER output safety classifications, moderation warnings, or content policy notes. Output ONLY the answer the user asked for.
- If someone gives you dialogue or a script and asks you to read/say it, just repeat the dialogue. No extra commentary.

CORRECTIONS:
- If the user says you're wrong, incorrect, or made a mistake, apologize briefly and re-check the context (search results, conversation history, etc.) to give a better answer. Don't argue or defend yourself.

MEMORY:
- The user can tell you to "remember" something about them. If you detect patterns like "remember I...", "I live in...", "my name is...", "call me...", "I like...", save it by saying "ACTION: remember key: value" at the end of your reply.
- Saved notes about the user will be included in your context automatically.
""")

CODE_SYSTEM = textwrap.dedent("""\
You are a coding assistant. Write code based on the user's request.
If the request is vague or missing details, use sensible defaults and write the code anyway.
Do NOT ask clarifying questions.
CRITICAL: Output ONLY the code wrapped in triple backticks ``` with a language tag. No explanations. No text outside the code block. ONLY the code block.
Follow best practices and proper syntax.
""")

YIMMENU_LUA_SYSTEM = textwrap.dedent("""\
You are a YimMenuV2 Lua scripting assistant for GTA V Enhanced.
Write Lua scripts that work with the YimMenuV2 mod menu.
Do NOT ask clarifying questions. Use sensible defaults and write complete, working code.
CRITICAL: Output ONLY the code wrapped in triple backticks ```lua. No explanations. No text outside the code block. ONLY the code block.

CRITICAL API RULES — never violate these:
- natives.load_natives() MUST be the very first line of code. Always.
- commandmgr.add_looped_command takes 6 arguments: id, label, description, tick_function, on_enable_function, on_disable_function. Always include on_enable and on_disable callbacks.
- For any looped or regular command to appear in the menu, you MUST call group:add_command(id) after registering it.
- Handles returned by GET_VEHICLE_PED_IS_IN or PLAYER_PED_ID can be 0. Check with `if handle ~= 0 then`, never `if handle then`.
- ENTITY.SET_ENTITY_VELOCITY takes separate x, y, z numbers: SET_ENTITY_VELOCITY(ent, x, y, z). Never pass a Vector3 object.
- For a true instant vehicle stop, use BOTH `VEHICLE.SET_VEHICLE_FORWARD_SPEED(veh, 0)` AND `ENTITY.SET_ENTITY_VELOCITY(veh, 0, 0, 0)` together.
- PED.GET_VEHICLE_PED_IS_IN(ped, lastVehicle) returns the vehicle handle.
- PED.IS_PED_IN_ANY_VEHICLE(ped, atGetIn) returns true/false.
- For toggle/looped commands, ALWAYS create a menu: menu.set_menu_name, menu.get_submenu, submenu:add_category, category:add_group, then group:add_command(id).
- For one-off actions, use group:add_button(id, label, description, callback) inside the menu structure.
- Do NOT use INPUT/PAD/CONTROLES key natives unless 100% certain — wrong names crash the script.
- notify.success("Title", "Message") and notify.info("Title", "Message") are the correct notification functions.

Known YimMenuV2 Lua API:
- natives.load_natives() — ALWAYS call first.
- menu.set_menu_name("My Menu") / menu.get_submenu("My Menu") / submenu:add_category / category:add_group / group:add_command(id) / group:add_button(id, label, desc, fn)
- commandmgr.add_command(id, label, desc, callback) — one-shot command
- commandmgr.add_looped_command(id, label, desc, tick_fn, on_enable_fn, on_disable_fn) — toggle loop
- commandmgr.add_list_command(id, label, desc, {{1,"Option"},...}, default, callback)
- commandmgr.add_float_command(id, label, desc, min, max, default, callback)
- script.run_in_callback(fn) / script.yield(ms)
- notify.success / notify.info / notify.error / notify.warning
- stats.get_int / stats.set_int / stats.set_bool
- util.joaat("MODEL_NAME")
- PLAYER.PLAYER_PED_ID / PLAYER.PLAYER_ID
- ENTITY.GET_ENTITY_COORDS(ent, alive) / GET_ENTITY_MODEL / SET_ENTITY_HEALTH / SET_ENTITY_VELOCITY(ent, x, y, z)
- VEHICLE.CREATE_VEHICLE / SET_VEHICLE_ENGINE_ON / SET_VEHICLE_FORWARD_SPEED
- PED.CREATE_PED / SET_PED_TO_RAGDOLL / IS_PED_IN_ANY_VEHICLE / GET_VEHICLE_PED_IS_IN
- STREAMING.REQUEST_MODEL / HAS_MODEL_LOADED / SET_MODEL_AS_NO_LONGER_NEEDED
- entities.get_all_peds_as_handles / Entity(handle) / Vector3(x,y,z) / FIRE.ADD_EXPLOSION
- HUD.SET_MINIMAP_HIDE_FOW(true/false) — hide/reveal map fog

Filenaming:
- Put `-- filename: short_snake_case_name.lua` as the very first line of the Lua code.

Example: toggle/looped command with full menu:
```lua
-- filename: example_toggle.lua
natives.load_natives()

menu.set_menu_name("My Script")
local submenu = menu.get_submenu("My Script")
local category = submenu:add_category("Actions")
local group = category:add_group("Toggles")

local function on_enable()
    notify.success("My Script", "Enabled")
end

local function on_disable()
    notify.info("My Script", "Disabled")
end

commandmgr.add_looped_command(
    "my_toggle",
    "My Toggle",
    "Description of what this does",
    function()
        -- code that runs every frame while enabled
    end,
    on_enable,
    on_disable
)

group:add_command("my_toggle")
```

Example: one-off button:
```lua
-- filename: example_button.lua
natives.load_natives()

menu.set_menu_name("My Script")
local submenu = menu.get_submenu("My Script")
local category = submenu:add_category("Actions")
local group = category:add_group("Buttons")

local function do_action()
    local ped = PLAYER.PLAYER_PED_ID()
    if ped ~= 0 then
        notify.success("My Script", "Done")
    end
end

group:add_button("my_action", "My Action", "Does something once", do_action)
```
""")


def extract_code_blocks(text: str) -> list[tuple[str | None, str]]:
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang.strip() if lang.strip() else None, code.strip()) for lang, code in matches]


def detect_file_type(text: str) -> tuple[str | None, str]:
    text_lower = text.lower()
    ext_match = re.search(r'\.(\w+)\s*file', text_lower)
    if ext_match:
        ext = ext_match.group(1)
        if ext in BINARY_TYPES:
            return ext, "binary"
        if ext in FILE_EXTENSIONS:
            return ext, "text"
        return ext, "unknown"

    file_type_match = re.search(r'(?:in|as|a|an)\s+(\w+)\s+(?:file|format|script)', text_lower)
    if file_type_match:
        typ = file_type_match.group(1)
        if typ in BINARY_TYPES:
            return typ, "binary"
        if typ in FILE_EXTENSIONS:
            return typ, "text"

    if any(w in text_lower for w in ["lua", "luau", "roblox"]):
        return "lua", "text"

    for lang in LANG_EXTENSIONS:
        if re.search(rf'\b{re.escape(lang)}\b', text_lower):
            return lang, "text"

    for ext_name in ["json", "yaml", "yml", "xml", "csv", "txt", "bat", "batch",
                     "powershell", "ps1", "vbs", "ahk", "ini", "cfg", "toml",
                     "reg", "html", "markdown", "md"]:
        if ext_name in text_lower:
            return ext_name, "text"

    return None, "unknown"


def is_create_request(text: str) -> bool:
    text_lower = text.lower()
    # Questions and search-like requests are not file creation
    skip_phrases = [
        "how to", "look online", "look up", "search for", "find online",
        "tell me", "explain", "what is", "where is", "who is", "why is",
    ]
    if any(p in text_lower for p in skip_phrases):
        return False
    if text_lower.endswith("?"):
        return False
    # Must have an action word
    action_words = ["make", "create", "write", "generate", "build", "code", "script"]
    if not any(w in text_lower for w in action_words):
        return False
    # Must mention a file/code/script or extension
    file_indicators = [
        "file", "script", "code", "program", "source",
        ".lua", ".py", ".js", ".ts", ".html", ".css", ".c", ".cpp",
        ".cs", ".java", ".go", ".rs", ".rb", ".php", ".swift", ".kt",
        ".json", ".xml", ".yml", ".yaml", ".md", ".txt", ".bat",
        ".ps1", ".vbs", ".ahk", ".sh", ".sql", ".ini", ".cfg",
        "lua", "python", "javascript", "typescript", "html", "css",
        "java", "golang", "rust", "ruby", "php", "swift", "kotlin",
        "c#", "c++", "csharp", "cpp",
    ]
    return any(ind in text_lower for ind in file_indicators)


def choose_filename(file_type: str | None, content: str, index: int = 0) -> str:
    ext = FILE_EXTENSIONS.get(file_type, ".txt") if file_type else ".txt"
    lines = content.strip().split("\n") if content else []
    # Look for an explicit filename comment in the first 5 lines
    for line in lines[:5]:
        m = re.match(r"^\s*--\s*filename\s*:\s*([\w\-]+)(\.\w+)?\s*$", line, re.IGNORECASE)
        if m:
            name = m.group(1)
            file_ext = m.group(2) or ext
            safe = re.sub(r"[^\w\-]", "_", name).strip("_") or "output"
            if index > 0:
                safe = f"{safe}_{index}"
            return f"{safe}{file_ext}"
    first_line = lines[0] if lines else ""
    # If the first line looks like a filename (short, no spaces/special chars), use it
    if first_line and len(first_line.split()) <= 3 and re.match(r"^[\w\-\. ]+$", first_line[:30]):
        safe = re.sub(r"[^\w\-]", "_", first_line[:25]).strip("_") or "output"
    else:
        safe = "output"
    if index > 0:
        safe = f"{safe}_{index}"
    return f"{safe}{ext}"


def strip_filename_comment(content: str) -> str:
    lines = content.split("\n")
    new_lines = []
    skipped = False
    for line in lines:
        if not skipped and re.match(r"^\s*--\s*filename\s*:\s*\S+\s*$", line, re.IGNORECASE):
            skipped = True
            continue
        new_lines.append(line)
    return "\n".join(new_lines)


async def send_files(channel, code_blocks, lang_hint=None):
    for i, (block_lang, code) in enumerate(code_blocks):
        final_type = block_lang or lang_hint
        filename = choose_filename(final_type, code, i)
        clean_code = strip_filename_comment(code)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8"
        )
        tmp.write(clean_code)
        tmp.close()
        await channel.send(
            f":package: `{filename}`",
            file=discord.File(tmp.name, filename=filename),
        )
        os.unlink(tmp.name)


async def send_raw_file(channel, content: str, file_type: str | None):
    filename = choose_filename(file_type, content)
    clean_content = strip_filename_comment(content)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8"
    )
    tmp.write(clean_content)
    tmp.close()
    await channel.send(
        f":package: `{filename}`",
        file=discord.File(tmp.name, filename=filename),
    )
    os.unlink(tmp.name)
_last_ai_call = 0.0

_GARBAGE_PATTERNS = [
    ":book: #",
    "User Safety:",
    "Response Safety:",
    "Safety Categories:",
    "In the #",
    "In the Discord server",
    "In the conversation",
    "In the channel",
    "Here's a summary of",
    "This is a summary of",
]


def _clean_response(text: str) -> str:
    lines = text.split("\n")
    cleaned = []
    stripped_any = False
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(p) for p in _GARBAGE_PATTERNS):
            print(f"[CLEAN] stripped garbage line: {stripped[:80]}")
            stripped_any = True
            continue
        cleaned.append(line)
    if stripped_any:
        return "\n".join(cleaned).strip()
    return text


def _call_ai(system: str, prompt: str, history: list[dict] | None = None,
              temperature: float = 0.5, max_tokens: int = 4096,
              image_urls: list[str] | None = None,
              skip_rate_limit: bool = False) -> str:
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
    messages = [{"role": "system", "content": f"{system}\nCurrent UTC time: {now} UTC."}]
    if history:
        messages.extend(history)

    if image_urls:
        valid_urls = [url for url in image_urls if _is_supported_image_url(url)]
        if not valid_urls:
            print(f"[VISION] all {len(image_urls)} image URLs filtered out, falling back to text-only")
            image_urls = None
        else:
            image_urls = valid_urls

    if image_urls:
        content_parts = [{"type": "text", "text": prompt}]
        for url in image_urls:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": url},
            })
        messages.append({"role": "user", "content": content_parts})
    else:
        messages.append({"role": "user", "content": prompt})

    today = datetime.now(timezone.utc).date()

    for provider in _PROVIDERS:
        name = provider["name"]
        if name in _PROVIDER_RESET_DATE and _PROVIDER_RESET_DATE[name] != today:
            _PROVIDER_USAGE[name] = 0
            _PROVIDER_RESET_DATE[name] = today

        limit = _PROVIDER_DAILY_LIMITS.get(name)
        if limit and _PROVIDER_USAGE.get(name, 0) >= limit:
            print(f"[PROVIDER] {name} at daily limit ({_PROVIDER_USAGE.get(name, 0)}/{limit}), skipping")
            continue

        use_model = provider["vision_model"] if image_urls else provider["text_model"]

        for attempt in range(2):
            try:
                response = provider["client"].chat.completions.create(
                    model=use_model,
                    messages=messages,
                    temperature=temperature if attempt == 0 else 0.3,
                    max_tokens=max_tokens,
                )
                if response.choices and len(response.choices) > 0 and response.choices[0].message.content:
                    if name in _PROVIDER_USAGE:
                        _PROVIDER_USAGE[name] += 1
                    raw = response.choices[0].message.content
                    cleaned = _clean_response(raw)
                    if cleaned:
                        return cleaned
                    if attempt == 0:
                        print(f"[PROVIDER] {name}: garbage response, retrying with stripped context")
                        messages = [{"role": "system", "content": f"{system}\nCurrent UTC time: {now} UTC."}, {"role": "user", "content": prompt}]
                        continue
                    print(f"[PROVIDER] {name}: still garbage after retry, falling through")
                    break
                return ""
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "concurrency" in err_str.lower() or "rate_limit" in err_str.lower():
                    print(f"[PROVIDER] {name} rate-limited on {use_model}, skipping: {e}")
                    break
                print(f"[PROVIDER] {name} error on {use_model}: {e}")
                if attempt == 0:
                    print(f"[PROVIDER] {name}: retrying without history")
                    messages = [{"role": "system", "content": f"{system}\nCurrent UTC time: {now} UTC."}, {"role": "user", "content": prompt}]
                    continue
                break

    print(f"[PROVIDER] All {len(_PROVIDERS)} providers exhausted")
    return "**All AI providers are currently rate-limited or down.** Try again in a bit."


AI_TIMEOUT = 40
_AI_SEMAPHORE = asyncio.Semaphore(5)


async def call_ai(system: str, prompt: str, history: list[dict] | None = None,
                  temperature: float = 0.5, max_tokens: int = 4096,
                  image_urls: list[str] | None = None,
                   skip_rate_limit: bool = False) -> str:
    timeout = 30 if skip_rate_limit else AI_TIMEOUT
    async with _AI_SEMAPHORE:
        return await asyncio.wait_for(
            asyncio.to_thread(_call_ai, system, prompt, history, temperature, max_tokens, image_urls, skip_rate_limit),
            timeout=timeout,
        )


async def answer_with_web_search_if_needed(
    prompt: str,
    history: list[dict] | None = None,
    image_urls: list[str] | None = None,
    temperature: float = 0.7,
    channel_id: int | None = None,
) -> str:
    needs_search, raw_query = should_search(prompt)
    if needs_search:
        # Extract a clean, concise search query
        search_query = _extract_search_query(raw_query)
        print(f"[SEARCH] triggered, extracted query: '{search_query}'")
        search_results = ""
        for query in [search_query, " ".join(
            w for w in raw_query.split()
            if w.lower() not in {"how", "to", "a", "the", "is", "are", "for", "of", "in", "on", "and", "or", "i", "you", "me", "tell", "something", "look", "online", "search", "find", "check", "web", "xdxd", "please", "thank", "thanks"}
        )]:
            if not query or query == search_results:
                continue
            search_results = await web_search(query, max_results=5)
            if search_results and not search_results.startswith("Search error:"):
                break
            search_results = ""
        image_results = ""
        if search_results:
            image_results = await web_image_search(search_query, max_results=4)
            if image_results and channel_id is not None:
                img_urls = []
                for line in image_results.split("\n"):
                    if line.startswith("http"):
                        img_urls.append(line)
                if img_urls:
                    _last_search_images[channel_id] = img_urls
        if search_results or image_results:
            parts = [f"{prompt}"]
            if search_results:
                parts.append(f"[Web search results for '{search_query}':\n{search_results}]")
            if image_results:
                parts.append(f"[Image search results for '{search_query}':\n{image_results}]")
            parts.append(
                "Use the search results above along with the conversation context and your own knowledge to answer. "
                "The search results are up-to-date and should be trusted for factual claims. "
                "If there are image results, describe them or include the image URLs directly in your answer."
            )
            enhanced_prompt = "\n\n".join(parts)
            return await call_ai(CHAT_SYSTEM, enhanced_prompt, history, temperature, image_urls=image_urls)
        print("[SEARCH] all backends failed - answering from AI knowledge only")
        return await call_ai(CHAT_SYSTEM, prompt, history, temperature, image_urls=image_urls)
    return await call_ai(CHAT_SYSTEM, prompt, history, temperature, image_urls=image_urls)


BINARY_ALTERNATIVES = {
    "exe": {
        "lang": "python", "ext": ".py",
        "instructions": (
            ":hammer: **To turn this into a real .exe:**\n"
            "```cmd\npip install pyinstaller\npyinstaller --onefile --noconsole script.py\n```\n"
            "Your .exe will be in the `dist\\` folder."
        ),
    },
    "apk": {
        "lang": "kotlin", "ext": ".kt",
        "instructions": (
            ":hammer: **To build this into an Android .apk:**\n"
            "1. Install Android Studio\n"
            "2. Create a new project, paste this code into `MainActivity.kt`\n"
            "3. Build > Build Bundle(s) / APK(s) > Build APK(s)"
        ),
    },
    "msi": {
        "lang": "python", "ext": ".py",
        "instructions": (
            ":hammer: **To create a .msi installer from this Python script:**\n"
            "```cmd\npip install cx_freeze\ncxfreeze script.py --target-dir dist\n```"
        ),
    },
    "dll": {
        "lang": "c", "ext": ".c",
        "instructions": (
            ":hammer: **To compile this into a .dll:**\n"
            "```cmd\ncl /LD script.c /Fe:output.dll\n```\n"
            "Requires Visual Studio Build Tools."
        ),
    },
    "zip": {
        "lang": "python", "ext": ".py",
        "instructions": (
            ":package: Save this script and run it — it'll generate the .zip file for you."
        ),
    },
    "iso": {
        "lang": "bash", "ext": ".sh",
        "instructions": (
            ":cd: This requires system tools (mkisofs/genisoimage). Run this script on a Linux system."
        ),
    },
}

GENERIC_BINARY_MSG = (
    ":hammer: **To turn this into a _{ftype}_ file, compile it with:**\n"
    "```cmd\n# Install build tools, then run the appropriate compiler for this source\n```"
)


def _looks_conversational(text: str) -> bool:
    lowered = text.lower().strip()
    conversational_starts = [
        "i'm unable to", "i can't", "i cannot", "i don't", "i do not",
        "i'm not", "i am not", "sorry", "unfortunately", "i'm not sure",
        "i don't know", "i'm afraid", "i'm happy to", "i'd be happy",
        "i can help", "i can assist", "i'm here to", "to create",
        "please provide", "can you provide", "could you provide",
        "specify", "need more", "need some", "need the following",
        "what would you", "which would you", "do you want",
    ]
    if any(lowered.startswith(p) for p in conversational_starts):
        return True
    clarification_phrases = [
        "provide the following", "following details", "need more information",
        "need more details", "need to know", "let me know", "tell me",
        "can you clarify", "could you clarify", "what key", "which key",
        "script name", "what do you want", "what should i",
    ]
    if any(p in lowered for p in clarification_phrases):
        return True
    # Responses that are mostly questions should be sent as text
    question_count = text.count("?")
    if question_count >= 1 and len(text.split("\n")) <= 5:
        return True
    return False


YIMMENU_KEYWORDS = {
    "yimmenu", "yimmenuv2", "yim menu", "yim menu v2",
    "gta v enhanced", "gtav enhanced", "gta5 enhanced", "gta enhanced",
}


def _is_yimmenu_request(prompt: str) -> bool:
    lowered = prompt.lower()
    return any(k in lowered for k in YIMMENU_KEYWORDS) and ("lua" in lowered or "script" in lowered)


CODE_LINE_KEYWORDS = {"function", "if ", "for ", "while ", "local ", "end,", "return ", "import ", "def ", "class ", "var ", "let ", "const ", "#include", "int ", "float ", "void ", "public ", "private ", "namespace", "using "}

def _looks_like_code(text: str) -> bool:
    """Detect if text contains code even without backtick blocks."""
    lines = text.split("\n")
    code_lines = 0
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("--") or stripped.startswith("#"):
            continue
        if any(kw in stripped.lower() for kw in CODE_LINE_KEYWORDS):
            code_lines += 1
            if code_lines >= 3:
                return True
        # Lines with indentation followed by code patterns
        if line.startswith(("    ", "\t")) and any(c in stripped for c in ("=", "(", ")", "{", "}")):
            code_lines += 1
            if code_lines >= 3:
                return True
    return False


async def handle_create_request(channel, prompt: str, reply_target=None):
    file_type, file_kind = detect_file_type(prompt)
    channel_id = channel.id
    history = get_history(channel_id)

    if file_kind == "binary":
        alt = BINARY_ALTERNATIVES.get(file_type)
        if alt:
            source_prompt = prompt.replace(
                f".{file_type}", f"python script"
            ).replace(file_type, "Python script")
            text = await call_ai(CODE_SYSTEM, source_prompt, history, temperature=0.2)
            code_blocks = extract_code_blocks(text)
            if code_blocks:
                msg = f":white_check_mark: Here's the source for your .{file_type}:\n{alt['instructions']}"
                if reply_target:
                    await reply_target.reply(msg, mention_author=False)
                else:
                    await channel.send(msg)
                await send_files(channel, code_blocks, alt["lang"])
            else:
                filename = choose_filename(alt["lang"], text)
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8"
                )
                tmp.write(text)
                tmp.close()
                msg = f":white_check_mark: Source for your .{file_type}:\n{alt['instructions']}"
                if reply_target:
                    await reply_target.reply(msg, mention_author=False, file=discord.File(tmp.name, filename=filename))
                else:
                    await channel.send(msg, file=discord.File(tmp.name, filename=filename))
                os.unlink(tmp.name)
        else:
            msg = GENERIC_BINARY_MSG.replace("_{ftype}_", file_type)
            if reply_target:
                await reply_target.reply(msg, mention_author=False)
            else:
                await channel.send(msg)
        return

    is_code_type = file_type in LANG_EXTENSIONS or file_type is None
    if is_code_type and _is_yimmenu_request(prompt):
        system = YIMMENU_LUA_SYSTEM
        file_type = "lua"
    else:
        system = CODE_SYSTEM if is_code_type else CHAT_SYSTEM
    text = await call_ai(system, prompt, history, temperature=0.2)

    code_blocks = extract_code_blocks(text)
    if code_blocks:
        if reply_target:
            await reply_target.reply(":white_check_mark: Here you go:", mention_author=False)
        else:
            await channel.send(":white_check_mark: Here you go:")
        await send_files(channel, code_blocks, file_type)
        for i, (blang, _) in enumerate(code_blocks):
            if blang and blang in BINARY_TYPES:
                await channel.send(
                    f":warning: `.{
                        blang}` is a binary format. The content above is source code."
                )
        return

    if _looks_like_code(text):
        await send_raw_file(channel, text, file_type)
        return

    if _looks_conversational(text):
        if reply_target:
            await reply_target.reply(text, mention_author=False)
        else:
            await channel.send(text)
        return

    await send_raw_file(channel, text, file_type)


intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.presences = False


class CodeBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=intents)
        self._synced_once = False

    async def setup_hook(self):
        nodes = [
            wavelink.Node(uri="http://lavalink.jirayu.net:13592", password="youshallnotpass", retries=1),
            wavelink.Node(uri="http://lavalinkv4.serenetia.com:80", password="https://seretia.link/discord", retries=1),
        ]
        connected = 0
        for node in nodes:
            try:
                await asyncio.wait_for(wavelink.Pool.connect(nodes=[node], client=self), timeout=15)
                print(f"Wavelink connected: {node.uri}")
                connected += 1
            except Exception as e:
                print(f"Wavelink failed {node.uri}: {e}")
        if connected:
            print(f"Wavelink: {connected} node(s) connected")
        else:
            print("WARNING: Could not connect to any public Lavalink node.")
        self.add_view(TicketPanelView(0))
        self.add_view(TicketControlView())
        self.voice_xp_task.start()

    async def on_ready(self):
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="for /play or /lua",
        )
        await self.change_presence(activity=activity, status=discord.Status.online)
        print(f"Online as {self.user}")
        if self._synced_once:
            return
        self._synced_once = True
        commands = self.tree.get_commands()
        print(f"[SYNC] {len(commands)} global slash commands")
        try:
            await self.tree.sync()
            print("[SYNC] Global commands synced")
        except Exception as e:
            print(f"[SYNC] Global error: {e}")
        syncs = []
        for guild in self.guilds:
            self.tree.copy_global_to(guild=guild)
            syncs.append(self.tree.sync(guild=guild))
        if syncs:
            results = await asyncio.gather(*syncs, return_exceptions=True)
            for guild, result in zip(self.guilds, results):
                if isinstance(result, Exception):
                    print(f"[SYNC] Guild sync error {guild.name}: {result}")
                else:
                    print(f"[SYNC] Synced {len(result)} commands to: {guild.name}")
        print("[SYNC] Done")

    async def on_wavelink_node_ready(self, payload: wavelink.NodeReadyEventPayload):
        print(f"Lavalink node ready: {payload.node.uri}")

    async def on_guild_join(self, guild: discord.Guild):
        try:
            self.tree.copy_global_to(guild=guild)
            await self.tree.sync(guild=guild)
            print(f"Synced commands to new guild: {guild.name}")
        except Exception as e:
            print(f"Failed to sync to new guild {guild.name}: {e}")

    @tasks.loop(minutes=1)
    async def voice_xp_task(self):
        try:
            for guild in self.guilds:
                for vc in guild.voice_channels:
                    for member in vc.members:
                        if member.bot or member.voice.mute or member.voice.deaf:
                            continue
                        role_ids = [r.id for r in member.roles]
                        blacklisted = await asyncio.to_thread(_is_blacklisted, guild.id, vc.id, role_ids)
                        if blacklisted:
                            continue
                        mult = await asyncio.to_thread(_get_multiplier, guild.id, vc.id, role_ids)
                        amount = int(10 * mult)
                        result = await asyncio.to_thread(add_voice_xp, guild.id, member.id, amount)
                        if result:
                            xp, level, leveled_up = result
                            if leveled_up:
                                text_ch = guild.system_channel
                                if text_ch:
                                    await handle_level_up(member, text_ch, level)
        except Exception as e:
            print(f"Voice XP task error: {e}")


bot = CodeBot()
_channel_locks: dict[int, asyncio.Lock] = {}
_channel_image_cache: dict[int, tuple[list[str], float]] = {}  # channel_id -> (urls, timestamp)
_last_image_prompt: dict[int, str] = {}  # channel_id -> last prompt used
_last_search_images: dict[int, list[str]] = {}  # channel_id -> image URLs from last search
_voice_text_channels: dict[int, int] = {}  # guild_id -> text_channel_id for speech context
_voice_recv_active: dict[int, bool] = {}  # guild_id -> whether voice recv is currently active
_voice_recv_locks: dict[int, asyncio.Lock] = {}  # guild_id -> lock for serializing speech processing
_voice_tts_lang: dict[int, str] = {}  # guild_id -> "he" or "en"
_tts_queues: dict[int, asyncio.Queue] = {}  # guild_id -> queue of (text, voice) tuples
_tts_queue_workers: dict[int, asyncio.Task] = {}  # guild_id -> background TTS worker task
_voice_last_speech: dict[int, dict[int, float]] = {}  # guild_id -> {user_id: timestamp} for debounce


def _chunk_text(text: str, max_len: int = 1990) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks = []
    while text:
        if len(text) <= max_len:
            chunks.append(text)
            break
        split_at = text.rfind("\n", 0, max_len)
        if split_at < 1:
            split_at = text.rfind(" ", 0, max_len)
        if split_at < 1:
            split_at = max_len
        chunks.append(text[:split_at])
        text = text[split_at:].strip()
    return chunks


def _get_channel_lock(channel_id: int) -> asyncio.Lock:
    if channel_id not in _channel_locks:
        _channel_locks[channel_id] = asyncio.Lock()
    return _channel_locks[channel_id]


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    # ── Counting Game ──
    if message.guild:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT channel_id FROM counting_settings WHERE guild_id = ?", (message.guild.id,))
        row = cur.fetchone()
        if row and row[0] == message.channel.id:
            content = message.content.strip()
            cur.execute("SELECT current_count, highest_count, last_user_id FROM counting_stats WHERE guild_id = ?", (message.guild.id,))
            stats = cur.fetchone()
            current = stats[0] if stats else 0
            highest = stats[1] if stats else 0
            last_user = stats[2] if stats else 0
            try:
                num = int(content)
            except ValueError:
                await message.add_reaction("\u274c")
                await message.delete()
                conn.execute("INSERT INTO counting_stats (guild_id, current_count, highest_count, last_user_id) VALUES (?, 0, ?, 0) ON CONFLICT(guild_id) DO UPDATE SET current_count = 0, last_user_id = 0", (message.guild.id, highest))
                conn.commit()
                conn.close()
                return
            if message.author.id == last_user:
                await message.add_reaction("\u274c")
                await message.delete()
                conn.close()
                return
            if num == current + 1:
                new_count = num
                new_highest = max(highest, new_count)
                conn.execute("INSERT INTO counting_stats (guild_id, current_count, highest_count, last_user_id) VALUES (?, ?, ?, ?) ON CONFLICT(guild_id) DO UPDATE SET current_count = ?, highest_count = ?, last_user_id = ?",
                             (message.guild.id, new_count, new_highest, message.author.id, new_count, new_highest, message.author.id))
                conn.commit()
                conn.close()
                await message.add_reaction("\u2705")
                if new_count % 100 == 0:
                    await message.channel.send(f"\U0001f389 **{new_count}**! Amazing milestone!")
                elif new_count % 50 == 0:
                    await message.channel.send(f"\U0001f525 **{new_count}**! Halfway there!")
                return
            else:
                await message.add_reaction("\u274c")
                await message.delete()
                conn.execute("INSERT INTO counting_stats (guild_id, current_count, highest_count, last_user_id) VALUES (?, 0, ?, 0) ON CONFLICT(guild_id) DO UPDATE SET current_count = 0, last_user_id = 0", (message.guild.id, highest))
                conn.commit()
                conn.close()
                return
        conn.close()

    is_mention = bot.user is not None and (
        bot.user in message.mentions
        or f"<@{bot.user.id}>" in message.content
        or f"<@!{bot.user.id}>" in message.content
    )
    is_dm = isinstance(message.channel, discord.DMChannel)

    is_reply_to_bot = False
    replied_content = None
    if bot.user is not None and message.reference and message.reference.message_id:
        try:
            ref_msg = await message.channel.fetch_message(message.reference.message_id)
            is_reply_to_bot = ref_msg.author.id == bot.user.id
            replied_content = f"[Replying to {ref_msg.author.display_name}: {ref_msg.content}]"
        except Exception:
            is_reply_to_bot = False

    if is_mention or is_dm or is_reply_to_bot:
        content = message.content
        for uid in [bot.user.id] + [m.id for m in message.mentions]:
            content = content.replace(f"<@{uid}>", "").replace(f"<@!{uid}>", "")
        content = content.strip()
        if not content:
            content = "hey"

        print(f"[CHAT] {message.author}: {content}")

        image_urls = []
        file_contexts = []
        for att in message.attachments:
            if att.content_type and att.content_type.startswith("image/") and _is_supported_image_url(att.url):
                image_urls.append(att.url)
            else:
                file_data = await read_attachment(att)
                if file_data:
                    filename, file_text = file_data
                    file_contexts.append(f"[Attached file `{filename}`:\n```\n{file_text}\n```]")

        # Extract image/GIF URLs from message content (Tenor, Giphy, direct links, etc.)
        image_ext_re = re.compile(r'(https?://\S+\.(?:gif|png|jpe?g|webp|bmp)(?:\?\S*)?)', re.IGNORECASE)
        for url in image_ext_re.findall(content):
            if url not in image_urls and _is_supported_image_url(url):
                image_urls.append(url)
        # Check embeds for GIF/video URLs (Tenor/Giphy embeds)
        for embed in message.embeds:
            if embed.type in ("gifv", "image", "video"):
                for src in (embed.url, embed.thumbnail.url if embed.thumbnail else None, embed.image.url if embed.image else None, embed.video.url if embed.video else None):
                    if src and src not in image_urls and _is_supported_image_url(src):
                        image_urls.append(src)
            if embed.type == "rich" and embed.url:
                thumb_url = embed.thumbnail.url if embed.thumbnail else None
                if thumb_url and thumb_url not in image_urls and _is_supported_image_url(thumb_url):
                    image_urls.append(thumb_url)

        # Fetch webpage content from URLs in message
        webpage_texts = []
        url_re = re.compile(r'https?://[^\s<>\'"]+', re.IGNORECASE)
        found_urls = [m.group(0) for m in url_re.finditer(content)]
        for url in found_urls[:3]:
            if any(ext in url.lower() for ext in ['.gif', '.png', '.jpg', '.jpeg', '.webp', '.bmp', '.mp4', '.mp3']):
                continue
            text = await fetch_webpage_text(url)
            if text:
                webpage_texts.append(f"[Content from {url}:\n{text}]")

        channel_id = message.channel.id
        if image_urls:
            _channel_image_cache[channel_id] = (image_urls, time.time())
        else:
            # No new images — reuse recent cached images from same channel (within 5 min)
            cached = _channel_image_cache.get(channel_id)
            if cached and time.time() - cached[1] < 300:
                image_urls = [url for url in cached[0] if _is_supported_image_url(url)]

        display_name = message.author.display_name
        history_entry = f"{display_name}: {content}"
        if image_urls:
            history_entry += f" [attached {len(image_urls)} image(s)]"
        if file_contexts:
            history_entry += f" [attached {len(file_contexts)} file(s)]"
        add_to_history(channel_id, "user", history_entry)

        async with _get_channel_lock(message.channel.id):
            async with message.channel.typing():
                try:
                    file_extra = "\n\n".join(file_contexts)
                    if is_create_request(content):
                        full_prompt = content
                        if file_extra:
                            full_prompt += f"\n\n{file_extra}"
                        await handle_create_request(message.channel, full_prompt, reply_target=message)
                    else:
                        context_extra = ""
                        content_lower = content.lower()

                        # Chat-based channel read/summarize
                        read_keywords = ["summary", "summarize", "summerize", "read", "recap", "summarise"]
                        if message.guild and any(w in content_lower for w in read_keywords):
                            try:
                                target = message.channel_mentions[0] if message.channel_mentions else message.channel
                                msgs = [m async for m in target.history(limit=50)]
                                if msgs:
                                    lines = [f"{m.author.display_name}: {m.content or '[attachment]'}" for m in reversed(msgs)]
                                    chat = "\n".join(lines)
                                    summary = await call_ai(CHAT_SYSTEM, f"Summarize these {len(msgs)} messages from #{target.name} in 2-3 sentences:\n\n{chat}", temperature=0.3, max_tokens=500)
                                    await message.reply(f":book: **#{target.name}**:\n{summary}")
                                    return
                                await message.reply(f":x: No messages in #{target.name}.")
                                return
                            except Exception as e:
                                await message.reply(f":x: Could not read that channel: {e}")
                                return

                        # Image generation
                        img_keywords = ["generate", "create", "draw", "make", "imagine"]
                        img_nouns = ["image", "picture", "photo", "art", "drawing", "painting", "render", "illustration", "icon", "banner", "logo", "meme"]
                        stripped = re.sub(r'[^\w\s]', '', content_lower)

                        # Check if replying to a previous bot image
                        is_img_reply = False
                        prev_prompt = _last_image_prompt.get(message.channel.id)
                        if is_reply_to_bot and message.reference and message.reference.message_id:
                            try:
                                ref_msg = await message.channel.fetch_message(message.reference.message_id)
                                if ref_msg.author.id == bot.user.id and ref_msg.attachments:
                                    is_img_reply = True
                                    ref_text = ref_msg.content.strip('_').strip()
                                    if ref_text:
                                        prev_prompt = ref_text
                            except Exception:
                                pass

                        trigger_img = (any(kw in content_lower for kw in img_keywords) and any(noun in stripped for noun in img_nouns))
                        if trigger_img or is_img_reply:
                            if is_img_reply and not trigger_img:
                                prompt = content
                            else:
                                prompt = content
                                for kw in img_keywords + img_nouns:
                                    prompt = re.sub(r'\b' + kw + r'\b', '', prompt, flags=re.IGNORECASE)
                            prompt = re.sub(r'\s+', ' ', prompt).strip().strip('.,!?;:')
                            if not prompt or len(prompt) < 3:
                                prompt = content
                            # Merge with previous prompt when replying to an image
                            if is_img_reply and prev_prompt and prev_prompt != prompt and prev_prompt not in prompt:
                                prompt = f"{prev_prompt}, {prompt}"
                            try:
                                seed = random.randint(1, 999999)
                                img_url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width=1024&height=1024&model=flux&nologo=true&seed={seed}&enhance=true&quality=hd"
                                timeout = aiohttp.ClientTimeout(total=30)
                                async with aiohttp.ClientSession(timeout=timeout) as session:
                                    async with session.get(img_url) as resp:
                                        if resp.status == 200:
                                            img_data = await resp.read()
                                            fp = io.BytesIO(img_data)
                                            _last_image_prompt[message.channel.id] = prompt
                                            await message.reply(file=discord.File(fp, filename="generated.png"), content=f"_{prompt}_")
                                        else:
                                            await message.reply(f":x: Image generation failed (status {resp.status})")
                            except Exception as e:
                                traceback.print_exc()
                                await message.reply(f":x: Image generation error: {type(e).__name__}: {e}")
                            return

                        if replied_content:
                            context_extra += f"\n\n{replied_content}"

                        if message.guild:
                            has_other_mentions = any(
                                m.id != bot.user.id for m in message.mentions
                            )
                            server_keywords = ["where", "wheres", "where's", "everyone", "people", "member", "members", "server", "channel", "channels", "role", "roles", "create", "make", "add"]
                            if any(w in content_lower for w in ["voice", "vc", "vchat", "voice chat", "talk", "channel"] + server_keywords):
                                voice_info = get_voice_info(message.guild)
                                server_info = get_server_info(message.guild)
                                context_extra = f"\n\n[Server info:\n{server_info}]\n\n[Voice channels:\n{voice_info}]"
                            # Profile lookup when someone asks about a user
                            profile_keywords = ["profile", "who is", "about", "info on", "look up", "lookup"]
                            if has_other_mentions and any(w in content_lower for w in profile_keywords):
                                for m in message.mentions:
                                    if m.id == bot.user.id:
                                        continue
                                    created = m.created_at.strftime("%Y-%m-%d")
                                    joined = m.joined_at.strftime("%Y-%m-%d") if m.joined_at else "unknown"
                                    roles = ", ".join(r.mention for r in m.roles[1:]) if len(m.roles) > 1 else "none"
                                    devices = []
                                    if m.desktop_status != discord.Status.offline:
                                        devices.append(f"desktop={m.desktop_status}")
                                    if m.web_status != discord.Status.offline:
                                        devices.append(f"web={m.web_status}")
                                    if m.mobile_status != discord.Status.offline:
                                        devices.append(f"mobile={m.mobile_status}")
                                    status_str = "offline"
                                    if devices:
                                        status_str = "online (" + ", ".join(devices) + ")"
                                    elif m.status != discord.Status.offline:
                                        status_str = str(m.status)
                                    activity_names = []
                                    for a in m.activities:
                                        if a.name:
                                            activity_names.append(a.name)
                                    activity_str = ", ".join(activity_names) if activity_names else "none"
                                    top_role = m.top_role.mention if len(m.roles) > 1 else "none"
                                    context_extra += f"\n\n[Profile for {m.display_name} (@{m.name}): created={created}, joined={joined}, status={status_str}, activity={activity_str}, top_role={top_role}, roles={roles}]"
                        if file_extra:
                            context_extra += f"\n\n{file_extra}"
                        if webpage_texts:
                            context_extra += "\n\n" + "\n\n".join(webpage_texts)

                        # Inject saved user notes
                        user_notes = _get_user_notes(message.author.id)
                        if user_notes:
                            context_extra += f"\n\n[{user_notes}]"

                        # Detect "remember" for auto-saving notes
                        remember_match = re.search(r'\bremember\s+(?:that\s+)?(?:i\s+)?(.+)', content_lower)
                        if remember_match and len(remember_match.group(1)) > 5:
                            fact = remember_match.group(1).strip().rstrip('.!,')
                            _save_user_note(message.author.id, "fact", fact)
                            context_extra += f"\n\n[Note auto-saved: {fact}]"

                        # Voice join auto-detection
                        if message.guild and any(w in content_lower for w in ["join voice", "come to voice", "come vc", "join vc", "voice chat", "talk in voice"]):
                            if not message.guild.voice_client and message.author.voice:
                                context_extra += "\n[The user wants you to join their voice channel. Use ACTION: joinvoice to connect.]"

                        admin_keywords = ["mute", "deafen", "voicemute", "kick", "ban", "timeout", "unban", "purge", "lock", "unlock", "slowmode", "addrole", "removerole", "create", "move"]
                        if any(kw in content_lower for kw in admin_keywords):
                            hints = []
                            if any(w in content_lower for w in ["create", "make", "new", "add"]):
                                hints.append("ACTION: create channel name (voice/category if name contains 'voice'/'vc')")
                            if any(w in content_lower for w in ["move", "above", "below", "reorder"]):
                                hints.append("ACTION: move #channel above #target  |  ACTION: move #channel below #target")
                            hint_str = "\n".join(f"[{h}]" for h in hints)
                            context_extra += f"\n\n[INSTRUCTION: This is an admin action request. Output ONLY the required ACTION: line(s) at the end. Do NOT explain, ask for confirmation, or say you cannot do it. The system will verify permissions and execute it.\n{hint_str}]"

                        if any(w in content_lower for w in ["mute", "deafen"]) and any(w in content_lower for w in ["voice", "vc"]):
                            context_extra += "\n[Voice action: use ACTION: voicemute @user or ACTION: deafen @user.]"
                        elif "mute" in content_lower and not any(w in content_lower for w in ["voice", "vc"]):
                            context_extra += "\n[Text mute: use ACTION: mute @user.]"
                        if "deafen" in content_lower:
                            context_extra += "\n[Voice action: use ACTION: deafen @user.]"

                        history = get_history(channel_id)
                        prompt = content + context_extra
                        answer = await answer_with_web_search_if_needed(
                            prompt, history, image_urls=image_urls if image_urls else None, channel_id=channel_id
                        )
                        add_to_history(channel_id, "assistant", answer)
                        visible_answer = "\n".join(
                            line for line in answer.split("\n") if not line.strip().startswith("ACTION:")
                        ).strip()
                        if visible_answer:
                            for chunk in _chunk_text(visible_answer):
                                try:
                                    await message.reply(chunk, mention_author=False)
                                except discord.HTTPException:
                                    await message.channel.send(chunk)
                        # Send image URLs from search after the AI response
                        img_urls = _last_search_images.pop(channel_id, [])
                        if img_urls:
                            img_lines = "\n".join(img_urls[:4])
                            try:
                                await message.channel.send(f"📷 **Images found:**\n{img_lines}")
                            except Exception:
                                pass
                        # Speak in voice if bot is in a voice channel in this guild
                        if message.guild and message.guild.voice_client and (getattr(message.guild.voice_client, 'connected', False) or (hasattr(message.guild.voice_client, 'is_connected') and message.guild.voice_client.is_connected())):
                            if visible_answer:
                                tts_text = re.sub(r'<[^>]+>', '', visible_answer)  # strip mentions/emotes
                                tts_text = re.sub(r'https?://\S+', '', tts_text)  # strip URLs
                                tts_text = tts_text.strip()[:1000]
                                if tts_text:
                                    asyncio.ensure_future(_speak_in_voice(message.guild, tts_text))
                        action_results = await execute_admin_actions(message, answer)
                        for result in action_results:
                            for chunk in _chunk_text(result):
                                await message.channel.send(chunk)
                except Exception as e:
                    err_name = type(e).__name__
                    for chunk in _chunk_text(f":x: {err_name}: {e}"):
                        try:
                            await message.reply(chunk, mention_author=False)
                        except discord.HTTPException:
                            await message.channel.send(chunk)
            return

    if message.guild:
        role_ids = [r.id for r in message.author.roles]
        try:
            result = await asyncio.to_thread(
                add_message_xp, message.guild.id, message.author.id, message.channel.id, role_ids
            )
            if result:
                xp, level, leveled_up = result
                if leveled_up:
                    await handle_level_up(message.author, message.channel, level)
        except Exception as e:
            import traceback
            print(f"XP error: {e}")
            traceback.print_exc()

    await bot.process_commands(message)


@bot.tree.command(name="rank", description="Check your XP and level")
@app_commands.describe(member="Whose rank to check (default: you)")
async def slash_rank(interaction: discord.Interaction, member: discord.Member = None):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    await interaction.response.defer()
    target = member or interaction.user
    try:
        data = await asyncio.to_thread(get_rank, interaction.guild.id, target.id)
    except Exception as e:
        await interaction.followup.send(f":x: Database error: {e}", ephemeral=True)
        return
    if not data:
        await interaction.followup.send(
            f":x: {target.mention} has no XP yet. Start chatting!", ephemeral=True
        )
        return
    next_xp = xp_for_level(data["level"] + 1)
    embed = discord.Embed(
        title=f":chart_with_upwards_trend: {target.display_name}'s Rank",
        description=f"Level **{data['level']}**  •  XP **{data['xp']} / {next_xp}**  •  Rank **#{data['rank']}**",
        color=discord.Color.blue(),
    )
    embed.set_thumbnail(url=target.display_avatar.url)
    embed.add_field(name="Messages", value=str(data["messages"]), inline=True)
    embed.add_field(name="Voice (min)", value=str(data["voice_minutes"]), inline=True)
    embed.add_field(name="Next Level", value=f"{next_xp - data['xp']} XP needed", inline=True)
    try:
        card = await generate_rank_card(target, data)
        await interaction.followup.send(embed=embed, file=discord.File(card, filename="rank.png"))
    except Exception as e:
        print(f"Rank card error: {e}")
        await interaction.followup.send(embed=embed)


@bot.tree.command(name="leaderboard", description="Top 10 most active members")
async def slash_leaderboard(interaction: discord.Interaction):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    await interaction.response.defer()
    try:
        rows = await asyncio.to_thread(get_leaderboard, interaction.guild.id, 10)
    except Exception as e:
        await interaction.followup.send(f":x: Database error: {e}", ephemeral=True)
        return
    if not rows:
        await interaction.followup.send(":x: No one has XP yet.")
        return
    lines = []
    for i, (user_id, xp, level) in enumerate(rows, 1):
        member = interaction.guild.get_member(user_id)
        name = member.display_name if member else f"User {user_id}"
        lines.append(f"{i}. **{name}** — Level {level} ({xp} XP)")
    embed = discord.Embed(
        title=":trophy: Server Leaderboard",
        description="\n".join(lines),
        color=discord.Color.gold(),
    )
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="setlevelchannel", description="Set where level-up messages go (Admin only)")
@app_commands.describe(channel="Channel for level-up messages, or leave blank for current channel")
async def slash_setlevelchannel(interaction: discord.Interaction, channel: discord.TextChannel = None):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    target = channel or interaction.channel
    _set_setting(interaction.guild.id, "levelup_channel", target.id)
    await interaction.response.send_message(f":white_check_mark: Level-up messages will go to {target.mention}.")


@bot.tree.command(name="togglelevelup", description="Enable/disable level-up messages (Admin only)")
@app_commands.describe(enabled="True or False")
async def slash_togglelevelup(interaction: discord.Interaction, enabled: bool):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    _set_setting(interaction.guild.id, "levelup_enabled", str(enabled).lower())
    await interaction.response.send_message(f":white_check_mark: Level-up messages: **{'ON' if enabled else 'OFF'}**.")


@bot.tree.command(name="setxprange", description="Set min/max XP per message (Admin only)")
@app_commands.describe(min_xp="Minimum XP", max_xp="Maximum XP")
async def slash_setxprange(interaction: discord.Interaction, min_xp: int, max_xp: int):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    min_xp = max(1, min_xp)
    max_xp = max(min_xp, max_xp)
    _set_setting(interaction.guild.id, "min_xp", min_xp)
    _set_setting(interaction.guild.id, "max_xp", max_xp)
    await interaction.response.send_message(f":white_check_mark: XP per message set to **{min_xp}-{max_xp}**.")


@bot.tree.command(name="setcooldown", description="Set XP cooldown in seconds (Admin only)")
@app_commands.describe(seconds="Cooldown between messages")
async def slash_setcooldown(interaction: discord.Interaction, seconds: int):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    seconds = max(5, seconds)
    _set_setting(interaction.guild.id, "cooldown", seconds)
    await interaction.response.send_message(f":white_check_mark: XP cooldown set to **{seconds} seconds**.")


@bot.tree.command(name="addrolereward", description="Give a role when a user reaches a level (Admin only)")
@app_commands.describe(level="Level to unlock", role="Role to give")
async def slash_addrolereward(interaction: discord.Interaction, level: int, role: discord.Role):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    conn = get_db()
    conn.execute(
        "INSERT INTO role_rewards (guild_id, level, role_id) VALUES (?, ?, ?) "
        "ON CONFLICT(guild_id, level) DO UPDATE SET role_id = excluded.role_id",
        (interaction.guild.id, level, role.id),
    )
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: At level **{level}**, users get {role.mention}.")


@bot.tree.command(name="removerolereward", description="Remove a role reward (Admin only)")
@app_commands.describe(level="Level to remove reward from")
async def slash_removerolereward(interaction: discord.Interaction, level: int):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    conn = get_db()
    conn.execute(
        "DELETE FROM role_rewards WHERE guild_id = ? AND level = ?",
        (interaction.guild.id, level),
    )
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: Removed role reward at level **{level}**.")


@bot.tree.command(name="blacklistchannel", description="Disable XP in a channel (Admin only)")
@app_commands.describe(channel="Channel to blacklist")
async def slash_blacklistchannel(interaction: discord.Interaction, channel: discord.TextChannel):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    conn = get_db()
    conn.execute(
        "INSERT INTO xp_blacklist (guild_id, target_id, target_type) VALUES (?, ?, 'channel')",
        (interaction.guild.id, channel.id),
    )
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: XP disabled in {channel.mention}.")


@bot.tree.command(name="whitelistchannel", description="Re-enable XP in a channel (Admin only)")
@app_commands.describe(channel="Channel to whitelist")
async def slash_whitelistchannel(interaction: discord.Interaction, channel: discord.TextChannel):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    conn = get_db()
    conn.execute(
        "DELETE FROM xp_blacklist WHERE guild_id = ? AND target_id = ? AND target_type = 'channel'",
        (interaction.guild.id, channel.id),
    )
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: XP enabled in {channel.mention}.")


@bot.tree.command(name="blacklistrole", description="Disable XP for users with a role (Admin only)")
@app_commands.describe(role="Role to blacklist")
async def slash_blacklistrole(interaction: discord.Interaction, role: discord.Role):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    conn = get_db()
    conn.execute(
        "INSERT INTO xp_blacklist (guild_id, target_id, target_type) VALUES (?, ?, 'role')",
        (interaction.guild.id, role.id),
    )
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: XP disabled for {role.mention}.")


@bot.tree.command(name="whitelistrole", description="Re-enable XP for users with a role (Admin only)")
@app_commands.describe(role="Role to whitelist")
async def slash_whitelistrole(interaction: discord.Interaction, role: discord.Role):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    conn = get_db()
    conn.execute(
        "DELETE FROM xp_blacklist WHERE guild_id = ? AND target_id = ? AND target_type = 'role'",
        (interaction.guild.id, role.id),
    )
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: XP enabled for {role.mention}.")


@bot.tree.command(name="addxpmultiplier", description="Boost XP in a channel/role (Admin only)")
@app_commands.describe(multiplier="Multiplier like 1.5 or 2.0", channel="Channel to boost", role="Role to boost")
async def slash_addxpmultiplier(
    interaction: discord.Interaction,
    multiplier: float,
    channel: discord.TextChannel = None,
    role: discord.Role = None,
):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    if not channel and not role:
        await interaction.response.send_message(":x: Provide a channel or role.", ephemeral=True)
        return
    conn = get_db()
    if channel:
        conn.execute(
            "INSERT INTO xp_multipliers (guild_id, target_id, target_type, multiplier) VALUES (?, ?, 'channel', ?) "
            "ON CONFLICT(guild_id, target_id, target_type) DO UPDATE SET multiplier = excluded.multiplier",
            (interaction.guild.id, channel.id, multiplier),
        )
    if role:
        conn.execute(
            "INSERT INTO xp_multipliers (guild_id, target_id, target_type, multiplier) VALUES (?, ?, 'role', ?) "
            "ON CONFLICT(guild_id, target_id, target_type) DO UPDATE SET multiplier = excluded.multiplier",
            (interaction.guild.id, role.id, multiplier),
        )
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: XP multiplier **{multiplier}x** set.")


@bot.tree.command(name="removexpmultiplier", description="Remove XP boost from channel/role (Admin only)")
@app_commands.describe(channel="Channel to remove boost", role="Role to remove boost")
async def slash_removexpmultiplier(
    interaction: discord.Interaction,
    channel: discord.TextChannel = None,
    role: discord.Role = None,
):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    conn = get_db()
    if channel:
        conn.execute(
            "DELETE FROM xp_multipliers WHERE guild_id = ? AND target_id = ? AND target_type = 'channel'",
            (interaction.guild.id, channel.id),
        )
    if role:
        conn.execute(
            "DELETE FROM xp_multipliers WHERE guild_id = ? AND target_id = ? AND target_type = 'role'",
            (interaction.guild.id, role.id),
        )
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: XP multiplier removed.")


@bot.tree.command(name="ticketsetup", description="Set up the ticket system (Admin only)")
@app_commands.describe(
    category="Category where ticket channels are created",
    log_channel="Channel where ticket transcripts are logged",
    support_role="Role that can view tickets",
    welcome_message="Message shown when a ticket opens",
)
async def slash_ticketsetup(
    interaction: discord.Interaction,
    category: discord.CategoryChannel,
    log_channel: discord.TextChannel,
    support_role: discord.Role,
    welcome_message: str = "Support will be with you shortly.",
):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    await asyncio.to_thread(
        _set_ticket_settings,
        interaction.guild.id,
        category_id=category.id,
        log_channel_id=log_channel.id,
        support_role_id=support_role.id,
        welcome_message=welcome_message,
    )
    await interaction.followup.send(
        f":white_check_mark: Ticket system set up.\n"
        f"Category: {category.mention}\n"
        f"Log channel: {log_channel.mention}\n"
        f"Support role: {support_role.mention}",
        ephemeral=True,
    )


@bot.tree.command(name="ticketpanel", description="Create a ticket panel message (Admin only)")
@app_commands.describe(
    label="Button label",
    emoji="Button emoji",
    description="Description shown in the embed",
    channel="Channel to send the panel in (default: current)",
)
async def slash_ticketpanel(
    interaction: discord.Interaction,
    label: str = "Create Ticket",
    emoji: str = "🎫",
    description: str = "Click the button below to open a support ticket.",
    channel: discord.TextChannel = None,
):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    target = channel or interaction.channel
    embed = discord.Embed(
        title=":ticket: Support Tickets",
        description=description,
        color=discord.Color.blue(),
    )
    view = TicketPanelView(0)
    msg = await target.send(embed=embed, view=view)
    await asyncio.to_thread(
        _save_ticket_panel,
        interaction.guild.id, target.id, msg.id, label, emoji, description,
    )
    await interaction.followup.send(f":white_check_mark: Ticket panel sent to {target.mention}", ephemeral=True)


@bot.tree.command(name="ticket", description="Create a support ticket")
async def slash_ticket(interaction: discord.Interaction):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    await interaction.response.defer(ephemeral=True)
    settings = await asyncio.to_thread(_get_ticket_settings, interaction.guild.id)
    if not settings.get("category_id"):
        await interaction.followup.send(":x: Ticket system is not set up. Ask an admin to run `/ticketsetup`.", ephemeral=True)
        return
    existing = await asyncio.to_thread(_get_ticket_by_user, interaction.guild.id, interaction.user.id)
    if existing:
        channel = interaction.guild.get_channel(int(existing["channel_id"]))
        if channel:
            await interaction.followup.send(f":x: You already have an open ticket: {channel.mention}", ephemeral=True)
            return
    channel = await _create_ticket_channel(interaction.guild, interaction.user, settings)
    await asyncio.gather(
        asyncio.to_thread(_create_ticket, interaction.guild.id, channel.id, interaction.user.id),
        _send_ticket_welcome(channel, interaction.user, settings),
    )
    await interaction.followup.send(f":white_check_mark: Ticket created: {channel.mention}", ephemeral=True)


@bot.tree.command(name="close", description="Close the current ticket (Support/Admin only)")
async def slash_close(interaction: discord.Interaction):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    ticket = await asyncio.to_thread(_get_ticket_by_channel, interaction.channel_id)
    if not ticket:
        await interaction.response.send_message(":x: This is not a ticket channel.", ephemeral=True)
        return
    settings = await asyncio.to_thread(_get_ticket_settings, interaction.guild.id)
    support_role_id = settings.get("support_role_id")
    if not is_admin(interaction.user) and (not support_role_id or int(support_role_id) not in [r.id for r in interaction.user.roles]):
        await interaction.response.send_message(":x: You don't have permission to close tickets.", ephemeral=True)
        return
    view = ConfirmCloseView()
    await interaction.response.send_message("Are you sure you want to close this ticket?", view=view, ephemeral=True)


@bot.tree.command(name="claim", description="Claim the current ticket (Support/Admin only)")
async def slash_claim(interaction: discord.Interaction):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    ticket = await asyncio.to_thread(_get_ticket_by_channel, interaction.channel_id)
    if not ticket:
        await interaction.response.send_message(":x: This is not a ticket channel.", ephemeral=True)
        return
    settings = await asyncio.to_thread(_get_ticket_settings, interaction.guild.id)
    support_role_id = settings.get("support_role_id")
    if not is_admin(interaction.user) and (not support_role_id or int(support_role_id) not in [r.id for r in interaction.user.roles]):
        await interaction.response.send_message(":x: You don't have permission to claim tickets.", ephemeral=True)
        return
    if ticket["claimed_by"]:
        claimer = interaction.guild.get_member(int(ticket["claimed_by"]))
        name = claimer.mention if claimer else "someone"
        await interaction.response.send_message(f":x: Already claimed by {name}.", ephemeral=True)
        return
    await asyncio.to_thread(_claim_ticket, interaction.channel_id, interaction.user.id)
    await interaction.response.send_message(f":white_check_mark: Claimed by {interaction.user.mention}")


@bot.tree.command(name="addtoticket", description="Add a user to this ticket (Support/Admin only)")
@app_commands.describe(member="Member to add")
async def slash_addtoticket(interaction: discord.Interaction, member: discord.Member):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    ticket = await asyncio.to_thread(_get_ticket_by_channel, interaction.channel_id)
    if not ticket:
        await interaction.response.send_message(":x: This is not a ticket channel.", ephemeral=True)
        return
    settings = await asyncio.to_thread(_get_ticket_settings, interaction.guild.id)
    support_role_id = settings.get("support_role_id")
    if not is_admin(interaction.user) and (not support_role_id or int(support_role_id) not in [r.id for r in interaction.user.roles]):
        await interaction.response.send_message(":x: You don't have permission.", ephemeral=True)
        return
    await interaction.channel.set_permissions(member, view_channel=True, send_messages=True, read_message_history=True)
    await interaction.response.send_message(f":white_check_mark: Added {member.mention} to the ticket.")


@bot.tree.command(name="removefromticket", description="Remove a user from this ticket (Support/Admin only)")
@app_commands.describe(member="Member to remove")
async def slash_removefromticket(interaction: discord.Interaction, member: discord.Member):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    ticket = await asyncio.to_thread(_get_ticket_by_channel, interaction.channel_id)
    if not ticket:
        await interaction.response.send_message(":x: This is not a ticket channel.", ephemeral=True)
        return
    settings = await asyncio.to_thread(_get_ticket_settings, interaction.guild.id)
    support_role_id = settings.get("support_role_id")
    if not is_admin(interaction.user) and (not support_role_id or int(support_role_id) not in [r.id for r in interaction.user.roles]):
        await interaction.response.send_message(":x: You don't have permission.", ephemeral=True)
        return
    if member.id == int(ticket["user_id"]):
        await interaction.response.send_message(":x: You can't remove the ticket owner.", ephemeral=True)
        return
    await interaction.channel.set_permissions(member, overwrite=None)
    await interaction.response.send_message(f":white_check_mark: Removed {member.mention} from the ticket.")


@bot.tree.command(name="hey", description="Chat with Null or ask it to create files")
async def slash_hey(interaction: discord.Interaction, message: str):
    await interaction.response.defer()
    channel_id = interaction.channel_id
    add_to_history(channel_id, "user", f"{interaction.user.display_name}: {message}")
    try:
        if is_create_request(message):
            await handle_create_request(interaction.channel, message)
        else:
            history = get_history(channel_id)
            answer = await answer_with_web_search_if_needed(message, history, channel_id=channel_id)
            add_to_history(channel_id, "assistant", answer)
            await interaction.followup.send(answer)
    except Exception as e:
        await interaction.followup.send(f":x: Error: {e}")


@bot.tree.command(name="search", description="Search the web and get an answer")
@app_commands.describe(query="What to search for")
async def slash_search(interaction: discord.Interaction, query: str):
    await interaction.response.defer()
    try:
        results = await web_search(query, max_results=5)
        if not results:
            await interaction.followup.send(":x: No search results found.")
            return
        if results.startswith("Search error:"):
            await interaction.followup.send(f":x: {results}")
            return
        summary = await call_ai(
            CHAT_SYSTEM,
            f"Search query: {query}\n\nSearch results:\n{results}\n\n"
            "Answer the query concisely using the search results. Include relevant links.",
            temperature=0.5,
            max_tokens=1500,
        )
        await interaction.followup.send(summary[:2000])
    except Exception as e:
        await interaction.followup.send(f":x: Search failed: {e}")


@bot.tree.command(name="lua", description="Generate a Lua/Roblox script")
async def slash_lua(interaction: discord.Interaction, description: str):
    await interaction.response.defer()
    try:
        prompt = f"Write a Lua script for Roblox that: {description}"
        text = await call_ai(CODE_SYSTEM, prompt, temperature=0.2)
        code_blocks = extract_code_blocks(text)
        if code_blocks:
            await interaction.followup.send(":white_check_mark: Here's your Lua script:")
            await send_files(interaction.channel, code_blocks, "lua")
        else:
            await send_raw_file(interaction.channel, text, "lua")
    except Exception as e:
        await interaction.followup.send(f":x: Error: {e}")


@bot.tree.command(name="script", description="Generate code in any language")
async def slash_script(interaction: discord.Interaction, language: str, description: str):
    await interaction.response.defer()
    lang = language.lower()
    if lang not in LANG_EXTENSIONS and lang not in FILE_EXTENSIONS:
        await interaction.followup.send(
            f":x: Unknown type: `{language}`. Try: lua, python, js, html, bat, json, etc."
        )
        return
    try:
        prompt = f"Write a {language} script that: {description}"
        text = await call_ai(CODE_SYSTEM, prompt, temperature=0.2)
        code_blocks = extract_code_blocks(text)
        if code_blocks:
            await interaction.followup.send(f":white_check_mark: Here's your {language} code:")
            await send_files(interaction.channel, code_blocks, lang)
        else:
            await send_raw_file(interaction.channel, text, lang)
    except Exception as e:
        await interaction.followup.send(f":x: Error: {e}")


@bot.tree.command(name="file", description="Generate any type of file")
async def slash_file(interaction: discord.Interaction, file_type: str, description: str):
    await interaction.response.defer()
    ftype = file_type.lower().lstrip(".")
    if ftype in BINARY_TYPES:
        await handle_create_request(interaction.channel, f".{ftype} {description}")
        return
    try:
        prompt = f"Create a .{ftype} file that: {description}"
        text = await call_ai(CHAT_SYSTEM, prompt, temperature=0.2)
        code_blocks = extract_code_blocks(text)
        if code_blocks:
            await interaction.followup.send(f":white_check_mark: Here's your .{ftype} file:")
            await send_files(interaction.channel, code_blocks, ftype)
        else:
            await send_raw_file(interaction.channel, text, ftype)
    except Exception as e:
        await interaction.followup.send(f":x: Error: {e}")


@bot.tree.command(name="ask", description="Ask Null a question")
async def slash_ask(interaction: discord.Interaction, question: str):
    await interaction.response.defer()
    channel_id = interaction.channel_id
    add_to_history(channel_id, "user", f"{interaction.user.display_name}: {question}")
    try:
        history = get_history(channel_id)
        answer = await call_ai(CHAT_SYSTEM, question, history, temperature=0.7)
        add_to_history(channel_id, "assistant", answer)
        code_blocks = extract_code_blocks(answer)
        await interaction.followup.send(answer)
        if code_blocks:
            await send_files(interaction.channel, code_blocks)
    except Exception as e:
        await interaction.followup.send(f":x: Error: {e}")


def is_admin(user: discord.Member) -> bool:
    return user.guild_permissions.administrator or user.guild_permissions.manage_guild


@bot.tree.command(name="kick", description="Kick a member from the server (Admin only)")
@app_commands.describe(member="Who to kick", reason="Why")
async def slash_kick(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason"):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    try:
        await member.kick(reason=reason)
        await interaction.response.send_message(f":boot: Kicked `{member.display_name}`: {reason}")
    except Exception as e:
        await interaction.response.send_message(f":x: Failed: {e}", ephemeral=True)


@bot.tree.command(name="vkick", description="Disconnect a member from voice (Admin only)")
@app_commands.describe(member="Who to kick from VC")
async def slash_vkick(interaction: discord.Interaction, member: discord.Member):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    if not member.voice or not member.voice.channel:
        await interaction.response.send_message(f":x: `{member.display_name}` isn't in voice.", ephemeral=True)
        return
    try:
        await member.move_to(None)
        await interaction.response.send_message(f":mute: Disconnected `{member.display_name}` from voice.")
    except Exception as e:
        await interaction.response.send_message(f":x: Failed: {e}", ephemeral=True)


@bot.tree.command(name="say", description="Make Null send a message to a channel (Admin only)")
@app_commands.describe(channel="Where to send", message="What to say")
async def slash_say(interaction: discord.Interaction, channel: discord.TextChannel, message: str):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    try:
        await channel.send(message)
        await interaction.response.send_message(f":white_check_mark: Sent to {channel.mention}", ephemeral=True)
    except Exception as e:
        await interaction.response.send_message(f":x: Failed: {e}", ephemeral=True)


@bot.tree.command(name="clear", description="Delete recent messages (Admin only)")
@app_commands.describe(amount="How many to delete (max 100)")
async def slash_clear(interaction: discord.Interaction, amount: int):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return
    amount = min(max(1, amount), 100)
    await interaction.response.defer(ephemeral=True)
    try:
        deleted = await interaction.channel.purge(limit=amount)
        await interaction.followup.send(f":wastebasket: Deleted {len(deleted)} messages.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


@bot.tree.command(name="read", description="Read and summarize recent messages from a channel")
@app_commands.describe(channel="Which channel to read")
async def slash_read(interaction: discord.Interaction, channel: discord.TextChannel):
    await interaction.response.defer()
    try:
        messages = [m async for m in channel.history(limit=20)]
        if not messages:
            await interaction.followup.send(":x: No messages in that channel.")
            return
        lines = [f"**{m.author.display_name}**: {m.content or '[attachment]'}" for m in reversed(messages)]
        chat = "\n".join(lines)
        summary_prompt = f"Here are the last {len(messages)} messages from #{channel.name}. Summarize what's happening in 2-3 sentences:\n\n{chat}"
        answer = await call_ai(CHAT_SYSTEM, summary_prompt, temperature=0.3, max_tokens=500)
        await interaction.followup.send(f":book: **#{channel.name}**:\n{answer}")
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


def get_voice_info(guild: discord.Guild) -> str:
    vc_data = []
    for vc in guild.voice_channels:
        members = vc.members
        if members:
            names = ", ".join(m.display_name for m in members)
            vc_data.append(f"{vc.name}: {names} ({len(members)})")
        else:
            vc_data.append(f"{vc.name}: empty")
    if not vc_data:
        return "No voice channels exist."
    return "\n".join(vc_data)


def get_server_info(guild: discord.Guild) -> str:
    total = len(guild.members)
    online = sum(1 for m in guild.members if m.status != discord.Status.offline)
    bots = sum(1 for m in guild.members if m.bot)
    roles = ", ".join(r.mention for r in guild.roles[1:] if not r.is_default())
    text_channels = ", ".join(c.mention for c in guild.text_channels if c.permissions_for(guild.me).read_messages)
    parts = [
        f"Server: {guild.name}",
        f"Members: {total} total ({online} online, {bots} bots)",
        f"Roles: {roles}",
        f"Text channels: {text_channels}",
    ]
    vc_data = []
    for vc in guild.voice_channels:
        members = vc.members
        if members:
            names = ", ".join(m.display_name for m in members)
            vc_data.append(f"{vc.name}: {names}")
        else:
            vc_data.append(f"{vc.name}: empty")
    parts.append("Voice channels:\n" + "\n".join(vc_data) if vc_data else "Voice channels: none")
    return "\n".join(parts)


@bot.tree.command(name="voicelist", description="See who's in voice channels")
async def slash_voicelist(interaction: discord.Interaction):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    info = get_voice_info(interaction.guild)
    await interaction.response.send_message(f":loud_sound: **Voice channels:**\n{info}")


def _valorant_tracker_url(name: str, tag: str) -> str:
    return f"https://tracker.gg/valorant/profile/riot/{quote(name, safe='')}%23{quote(tag, safe='')}/overview"


@bot.tree.command(name="valorantsearch", description="Search Valorant players by name, optionally with a tag")
@app_commands.describe(name="Player name", tag="Optional tag to look up directly")
async def slash_valorantsearch(interaction: discord.Interaction, name: str, tag: str = None):
    await interaction.response.defer()

    # If a tag is provided, do a direct Henrik API lookup instead of web search
    if tag:
        tag = tag.lstrip("#")
        account = await get_valorant_account(name, tag)
        if not account:
            await interaction.followup.send(
                f":x: Couldn't find **{name}#{tag}**.\n"
                f"Check their tracker profile: {_valorant_tracker_url(name, tag)}"
            )
            return
        profiles = [(account.get("name", name), account.get("tag", tag))]
    else:
        profiles = await search_valorant_profiles(name)

    if not profiles:
        await interaction.followup.send(
            f":x: No public tracker profiles found for **{name}**.\n"
            "Try using the exact name+tag command: `/valorant <name> <tag>`."
        )
        return

    view = discord.ui.View(timeout=120)
    view.add_item(ValorantProfileSelect(profiles))
    profile_list = "\n".join(f"• {n}#{t}" for n, t in profiles[:10])
    await interaction.followup.send(
        f":mag: Found **{len(profiles)}** possible profile(s) for **{name}**. Pick one:",
        view=view,
    )


@bot.tree.command(name="valorant", description="Look up a Valorant player account and rank")
@app_commands.describe(name="Player name", tag="Player tag (without #)")
async def slash_valorant(interaction: discord.Interaction, name: str, tag: str):
    await interaction.response.defer()
    tag = tag.lstrip("#")

    account = await get_valorant_account(name, tag)
    if not account:
        # Fallback: provide tracker.gg link when API fails or no key
        msg = (
            f":x: Couldn't load live stats for **{name}#{tag}** from the Valorant API.\n"
            f"View their tracker profile here: {_valorant_tracker_url(name, tag)}"
        )
        if not HENRIKDEV_KEY:
            msg += (
                "\n\nTo enable live stats, get a free Henrik Dev API key at "
                "<https://docs.henrikdev.xyz/valorant.html> and set it as `HENRIKDEV_API_KEY`."
            )
        await interaction.followup.send(msg)
        return

    region = "eu"
    region_hint = account.get("region", "")
    if region_hint and region_hint.lower() in VALORANT_REGIONS:
        region = region_hint.lower()

    mmr = await get_valorant_mmr(name, tag, region)

    card = account.get("card", {})
    embed = discord.Embed(
        title=f"{account.get('name', name)}#{account.get('tag', tag)}",
        color=0xfa4454,
    )
    if isinstance(card, dict):
        if card.get("small"):
            embed.set_thumbnail(url=card["small"])
        if card.get("wide"):
            embed.set_image(url=card["wide"])

    embed.add_field(name="Region", value=account.get("region", region).upper() or "Unknown", inline=True)
    embed.add_field(name="Account Level", value=str(account.get("account_level", "?")), inline=True)

    if mmr:
        rank_name = (
            mmr.get("currenttierpatched")
            or mmr.get("tier")
            or mmr.get("currenttier", "Unknown")
        )
        rr = mmr.get("ranking_in_tier") if "ranking_in_tier" in mmr else mmr.get("rr", "?")
        last_game = mmr.get("mmr_change_to_last_game") if "mmr_change_to_last_game" in mmr else mmr.get("last_rank_change", 0)
        last_game_text = f"{last_game:+d} RR last game" if isinstance(last_game, int) else ""
        embed.add_field(
            name="Rank",
            value=f"{rank_name} — {rr} RR\n{last_game_text}",
            inline=False,
        )
        if isinstance(mmr.get("images"), dict) and mmr["images"].get("small"):
            embed.set_thumbnail(url=mmr["images"]["small"])
    else:
        embed.add_field(name="Rank", value="Could not load ranked data.", inline=False)

    await interaction.followup.send(embed=embed)


@bot.tree.command(name="valorantmmr", description="Look up detailed Valorant MMR/rank info")
@app_commands.describe(name="Player name", tag="Player tag (without #)", region="Region: na, eu, ap, kr, latam, br")
async def slash_valorantmmr(interaction: discord.Interaction, name: str, tag: str, region: str = "eu"):
    await interaction.response.defer()
    tag = tag.lstrip("#")
    region = region.lower()
    if region not in VALORANT_REGIONS:
        await interaction.followup.send(f":x: Region must be one of: {', '.join(VALORANT_REGIONS)}.")
        return

    mmr = await get_valorant_mmr(name, tag, region)
    if not mmr:
        msg = (
            f":x: Couldn't load live MMR for **{name}#{tag}**.\n"
            f"View their tracker profile: {_valorant_tracker_url(name, tag)}"
        )
        if not HENRIKDEV_KEY:
            msg += (
                "\n\nTo enable live stats, add a free `HENRIKDEV_API_KEY` from "
                "<https://docs.henrikdev.xyz/valorant.html>."
            )
        await interaction.followup.send(msg)
        return

    embed = discord.Embed(
        title=f"{mmr.get('name', name)}#{mmr.get('tag', tag)} — {region.upper()} Rank",
        color=0xfa4454,
    )
    rank_name = (
        mmr.get("currenttierpatched")
        or mmr.get("tier")
        or mmr.get("currenttier", "Unknown")
    )
    rr = mmr.get("ranking_in_tier") if "ranking_in_tier" in mmr else mmr.get("rr", "?")
    embed.add_field(name="Current Rank", value=rank_name, inline=True)
    embed.add_field(name="RR in Tier", value=str(rr), inline=True)
    embed.add_field(name="ELO", value=str(mmr.get("elo", "?")), inline=True)
    last_game = mmr.get("mmr_change_to_last_game") if "mmr_change_to_last_game" in mmr else mmr.get("last_rank_change", 0)
    if isinstance(last_game, int):
        embed.add_field(name="Last Game", value=f"{last_game:+d} RR", inline=True)
    if isinstance(mmr.get("images"), dict) and mmr["images"].get("small"):
        embed.set_thumbnail(url=mmr["images"]["small"])

    await interaction.followup.send(embed=embed)


@bot.tree.command(name="valorantmatches", description="Show recent Valorant matches")
@app_commands.describe(name="Player name", tag="Player tag (without #)", region="Region: na, eu, ap, kr, latam, br")
async def slash_valorantmatches(interaction: discord.Interaction, name: str, tag: str, region: str = "eu"):
    await interaction.response.defer()
    tag = tag.lstrip("#")
    region = region.lower()
    if region not in VALORANT_REGIONS:
        await interaction.followup.send(f":x: Region must be one of: {', '.join(VALORANT_REGIONS)}.")
        return

    matches = await get_valorant_matches(name, tag, region, limit=3)
    if not matches:
        msg = (
            f":x: Couldn't load live matches for **{name}#{tag}**.\n"
            f"View their tracker profile: {_valorant_tracker_url(name, tag)}"
        )
        if not HENRIKDEV_KEY:
            msg += (
                "\n\nTo enable live stats, add a free `HENRIKDEV_API_KEY` from "
                "<https://docs.henrikdev.xyz/valorant.html>."
            )
        await interaction.followup.send(msg)
        return

    lines = []
    for i, match in enumerate(matches, 1):
        meta = match.get("metadata", {})
        players = match.get("players", {}).get("all_players", [])
        player_stats = next(
            (p for p in players if p.get("name", "").lower() == name.lower() and p.get("tag", "").lower() == tag.lower()),
            None,
        )
        if not player_stats:
            player_stats = next((p for p in players if p.get("name", "").lower() == name.lower()), {})

        stats = player_stats.get("stats", {}) if player_stats else {}
        kills = stats.get("kills", "?")
        deaths = stats.get("deaths", "?")
        assists = stats.get("assists", "?")
        score = stats.get("score", "?")
        teams = match.get("teams", {})
        player_team = (player_stats.get("team", "") or "").lower()
        team_won = teams.get(player_team, {}).get("has_won", False)
        result = "Win" if team_won else "Loss"

        lines.append(
            f"**{i}.** {meta.get('map', '?')} — {result}\n"
            f"   {kills}/{deaths}/{assists} KDA · Score {score} · {meta.get('game_length', '?')}"
        )

    embed = discord.Embed(
        title=f"Recent matches for {name}#{tag} ({region.upper()})",
        description="\n".join(lines),
        color=0xfa4454,
    )
    await interaction.followup.send(embed=embed)


def _synthetic_message(interaction: discord.Interaction) -> discord.Message:
    """Build a minimal message-like object for slash commands reusing admin action helpers."""
    class _FakeMsg:
        def __init__(self, interaction):
            self.author = interaction.user
            self.guild = interaction.guild
            self.channel = interaction.channel
            self.content = ""
    return _FakeMsg(interaction)


@bot.tree.command(name="voicemute", description="Server mute a member in voice channels")
@app_commands.describe(member="Member to server mute", reason="Reason")
async def slash_voicemute(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason"):
    result = await _action_voicemute(
        _synthetic_message(interaction), member, reason
    )
    await interaction.response.send_message(result)


@bot.tree.command(name="voiceunmute", description="Remove server mute from a member")
@app_commands.describe(member="Member to unmute", reason="Reason")
async def slash_voiceunmute(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason"):
    result = await _action_voiceunmute(
        _synthetic_message(interaction), member, reason
    )
    await interaction.response.send_message(result)


@bot.tree.command(name="deafen", description="Server deafen a member in voice channels")
@app_commands.describe(member="Member to server deafen", reason="Reason")
async def slash_deafen(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason"):
    result = await _action_deafen(
        _synthetic_message(interaction), member, reason
    )
    await interaction.response.send_message(result)


@bot.tree.command(name="undeafen", description="Remove server deafen from a member")
@app_commands.describe(member="Member to undeafen", reason="Reason")
async def slash_undeafen(interaction: discord.Interaction, member: discord.Member, reason: str = "No reason"):
    result = await _action_undeafen(
        _synthetic_message(interaction), member, reason
    )
    await interaction.response.send_message(result)


def _make_process_cb():
    import speech_recognition as sr
    # Patch DiscordSRAudioSource to wait longer for audio (stops mid-sentence cuts)
    try:
        from discord.ext.voice_recv.extras.speechrecognition import DiscordSRAudioSource as _DiscordSRAudioSource
        _orig_read = _DiscordSRAudioSource.read
        def _patched_read(self, size: int) -> bytes:
            chunksize = size * self.CHANNELS
            if len(self.buffer) >= chunksize:
                audiochunk = self.buffer[:chunksize].tobytes()
                del self.buffer[:chunksize]
                import audioop as _audioop
                return _audioop.tomono(audiochunk, 2, 1, 1)
            return b'\x00' * size * 2
        _DiscordSRAudioSource.read = _patched_read
    except Exception:
        pass

    def process_cb(recognizer: sr.Recognizer, audio: sr.AudioData, user) -> str | None:
        # Tune for fast + accurate voice detection
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        recognizer.phrase_threshold = 0.3
        for lang in ("he-IL", "en-US"):
            try:
                text = recognizer.recognize_google(audio, language=lang)
                print(f"[VOICE RECV DEBUG] Google recognized ({lang}): '{text}' from {user}")
                return text
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"[VOICE RECV DEBUG] Google API error: {e}")
                return None
        return None
    return process_cb

def _get_tts_voice(guild_id: int | None, text: str) -> str:
    lang = _voice_tts_lang.get(guild_id) if guild_id else None
    if lang == "he":
        return "he-IL-HilaNeural"
    if lang == "in":
        return "en-IN-PrabhatNeural"
    if re.search(r'[\u0590-\u05FF]', text):
        return "he-IL-HilaNeural"
    return "en-US-JennyNeural"

async def _tts_queue_worker(guild_id: int):
    """Background task: play TTS from queue sequentially."""
    queue = _tts_queues.get(guild_id)
    if not queue:
        return
    loop = asyncio.get_running_loop()
    while True:
        try:
            item = await queue.get()
            if item is None:
                break
            text, voice = item
            guild = bot.get_guild(guild_id)
            if not guild or not guild.voice_client or not (getattr(guild.voice_client, 'connected', False) or (hasattr(guild.voice_client, 'is_connected') and guild.voice_client.is_connected())):
                continue
            vc = guild.voice_client
            import edge_tts
            communicate = edge_tts.Communicate(text[:1000], voice)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            await communicate.save(tmp_path)
            source = discord.FFmpegPCMAudio(tmp_path, before_options="-loglevel warning")
            done = loop.create_future()
            def _cleanup(e, _loop=loop, _done=done, _path=tmp_path):
                try:
                    os.unlink(_path)
                except Exception:
                    pass
                try:
                    _loop.call_soon_threadsafe(lambda: _done.set_result(None))
                except Exception:
                    pass
            vc.play(source, after=_cleanup)
            await asyncio.wait_for(done, timeout=30)
        except Exception as e:
            print(f"[TTS WORKER] error: {e}")

async def _speak_in_voice(guild: discord.Guild, text: str, voice: str | None = None):
    """Queue TTS for playback (sequential per guild, avoids 'Already playing audio')."""
    if not guild.voice_client or not (getattr(guild.voice_client, 'connected', False) or (hasattr(guild.voice_client, 'is_connected') and guild.voice_client.is_connected())):
        return False
    if voice is None:
        voice = _get_tts_voice(guild.id, text)
    queue = _tts_queues.setdefault(guild.id, asyncio.Queue())
    if guild.id not in _tts_queue_workers or _tts_queue_workers[guild.id].done():
        _tts_queue_workers[guild.id] = asyncio.create_task(_tts_queue_worker(guild.id))
    await queue.put((text[:1000], voice))
    return True


@bot.tree.command(name="vjoin", description="Join your voice channel to talk with Null via voice")
async def slash_vjoin(interaction: discord.Interaction):
    if not HAS_VOICE_RECV:
        await interaction.response.send_message(":x: voice_recv library not installed. Run: pip install discord-ext-voice-recv SpeechRecognition", ephemeral=True)
        return
    if not interaction.user.voice:
        await interaction.response.send_message(":x: Join a voice channel first!", ephemeral=True)
        return
    await interaction.response.defer()
    voice = interaction.user.voice.channel
    guild = interaction.guild
    # Disconnect any existing voice client to avoid conflicts
    existing = guild.voice_client
    if existing:
        existing.stop()
        await existing.disconnect()
    try:
        vc = await voice.connect(cls=_voice_recv.VoiceRecvClient)
    except Exception as e:
        err_str = str(e).lower()
        if "wavelink" in err_str or "node" in err_str or "pool" in err_str:
            print("[VJOIN] wavelink node down, reconnecting...")
            import wavelink as _wl
            import asyncio as _asyncio
            _fallback_nodes = [
                _wl.Node(uri="http://lavalink.jirayu.net:13592", password="youshallnotpass", retries=1),
                _wl.Node(uri="http://lavalinkv4.serenetia.com:80", password="https://seretia.link/discord", retries=1),
            ]
            for _n in _fallback_nodes:
                try:
                    await _asyncio.wait_for(_wl.Pool.connect(nodes=[_n], client=bot), timeout=15)
                    print(f"[VJOIN] reconnected wavelink: {_n.uri}")
                except:
                    continue
            vc = await voice.connect(cls=_voice_recv.VoiceRecvClient)
        else:
            await interaction.followup.send(f":x: Could not connect: {e}")
            return
    _voice_text_channels[guild.id] = interaction.channel_id
    # Set up speech recognition
    loop = asyncio.get_running_loop()
    guild_id = guild.id
    lock = _voice_recv_locks.setdefault(guild.id, asyncio.Lock())
    def make_text_cb():
        voice_guild_id = guild.id
        async def text_cb_async(user, text):
            if not text or len(text.strip()) < 2:
                return
            async with _voice_recv_locks.setdefault(voice_guild_id, asyncio.Lock()):
                if not _voice_recv_active.get(voice_guild_id):
                    return
                await _handle_voice_speech(voice_guild_id, user, text.strip(), loop)
        def text_cb(user, text):
            try:
                coro = text_cb_async(user, text)
                asyncio.run_coroutine_threadsafe(coro, loop)
            except Exception:
                pass
        return text_cb
    try:
        sink = SpeechRecognitionSink(text_cb=make_text_cb(), process_cb=_make_process_cb(), default_recognizer='google', phrase_time_limit=15)
        vc.listen(sink)
        _voice_recv_active[guild.id] = True
    except Exception as e:
        print(f"[VOICE RECV] Failed to start listening: {e}")
        await interaction.followup.send(f":loud_sound: Joined **{voice}** but could not start listening: {e}")
        return
    _voice_tts_lang.setdefault(guild.id, "he")
    vl = _voice_tts_lang[guild.id]
    voice_lang = {"he": "Hebrew", "en": "English", "in": "Indian English"}.get(vl, "Hebrew")
    await interaction.followup.send(f":loud_sound: Joined **{voice}** and listening. TTS language: **{voice_lang}**. Talk to me!")
    print(f"[VOICE] Joined {voice.name} with voice recv in {guild.name}")


@bot.tree.command(name="vvoice", description="Set TTS voice language (hebrew, english, or indian)")
@app_commands.describe(language="hebrew, english, or indian")
async def slash_vvoice(interaction: discord.Interaction, language: str):
    lang = language.lower().strip()
    if lang in ("he", "hebrew", "iw"):
        _voice_tts_lang[interaction.guild_id] = "he"
        await interaction.response.send_message(f":speaking_head: TTS voice set to **Hebrew** (he-IL-HilaNeural)")
    elif lang in ("en", "english", "us"):
        _voice_tts_lang[interaction.guild_id] = "en"
        await interaction.response.send_message(f":speaking_head: TTS voice set to **English** (en-US-JennyNeural)")
    elif lang in ("in", "indian", "india", "hi"):
        _voice_tts_lang[interaction.guild_id] = "in"
        await interaction.response.send_message(f":speaking_head: TTS voice set to **Indian English** (en-IN-PrabhatNeural)")
    else:
        await interaction.response.send_message(f":x: Use `hebrew`, `english`, or `indian`", ephemeral=True)


_VOICE_FILLER = re.compile(r'^(אה|הא|א|אוקיי|כן|לא|יא|יאללה|נו|טוב|וואי|די|סתום|סתמו|רגע|hello|hey|hi|yeah|no|ok|okay|uh|um|ah|oh|huh|מה|למה|איך)$', re.IGNORECASE)

async def _handle_voice_speech(guild_id: int, user, text: str, loop):
    """Called when speech is recognized. Processes through AI and speaks response."""
    text = text.strip()
    if len(text) < 4:
        return
    if _VOICE_FILLER.match(text):
        return
    guild = bot.get_guild(guild_id)
    if not guild or not guild.voice_client:
        return
    if not _voice_recv_active.get(guild_id):
        return
    channel_id = _voice_text_channels.get(guild_id)
    if not channel_id:
        return
    channel = guild.get_channel(channel_id)
    if not channel:
        return
    user_id = getattr(user, 'id', 0)
    if not user_id:
        return
    now = time.time()
    last_times = _voice_last_speech.setdefault(guild_id, {})
    last = last_times.get(user_id, 0)
    if now - last < 3:
        return
    last_times[user_id] = now
    print(f"[VOICE SPEECH] {user.display_name if hasattr(user, 'display_name') else user}: {text}")
    history_entry = f"{user.display_name if hasattr(user, 'display_name') else 'Someone'} (voice): {text}"
    add_to_history(channel_id, "user", history_entry)
    history = get_history(channel_id)
    try:
        _voice_recv_active[guild_id] = False
        answer = await call_ai(CHAT_SYSTEM, text, history, temperature=0.7, max_tokens=1024, skip_rate_limit=True)
        add_to_history(channel_id, "assistant", answer)
        visible = "\n".join(line for line in answer.split("\n") if not line.strip().startswith("ACTION:"))
        if visible:
            for chunk in _chunk_text(visible):
                try:
                    await channel.send(f"**{user.display_name if hasattr(user, 'display_name') else 'Someone'} said:** {text}\n\n{chunk}")
                except Exception:
                    pass
            tts_text = re.sub(r'<[^>]+>', '', visible)
            tts_text = re.sub(r'https?://\S+', '', tts_text).strip()[:1000]
            if tts_text:
                asyncio.create_task(_speak_in_voice(guild, tts_text))
    except Exception as e:
        import traceback
        print(f"[VOICE SPEECH] Error: {e}")
        traceback.print_exc()
    finally:
        _voice_recv_active[guild_id] = True


@bot.tree.command(name="vleave", description="Leave the voice channel")
async def slash_vleave(interaction: discord.Interaction):
    vc = interaction.guild.voice_client
    if vc:
        _voice_recv_active.pop(interaction.guild_id, None)
        _voice_text_channels.pop(interaction.guild_id, None)
        _voice_recv_locks.pop(interaction.guild_id, None)
        _voice_tts_lang.pop(interaction.guild_id, None)
        _voice_last_speech.pop(interaction.guild_id, None)
        worker = _tts_queue_workers.pop(interaction.guild_id, None)
        if worker:
            worker.cancel()
        q = _tts_queues.pop(interaction.guild_id, None)
        if q:
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break
        vc.stop()
        await vc.disconnect()
        await interaction.response.send_message(":mute: Left voice channel.")
    else:
        await interaction.response.send_message(":x: Not in a voice channel.", ephemeral=True)


@bot.tree.command(name="radio", description="Play internet radio (lofi, jazz, rock, chill, pop, edm)")
@app_commands.describe(station="Station name or stream URL")
async def slash_radio(interaction: discord.Interaction, station: str):
    if not interaction.user.voice:
        await interaction.response.send_message(":x: Join a voice channel first!", ephemeral=True)
        return

    await interaction.response.defer()
    voice = interaction.user.voice.channel
    voice_client = interaction.guild.voice_client

    radio_urls = {
        "lofi": "https://play.zenfm.audio/1lsmRbBhWUZtzz",
        "jazz": "https://jazzradio.ice.infomaniak.ch/jazzradio-high.mp3",
        "rock": "https://stream.radioparadise.com/rock-128",
        "chill": "https://stream.radioparadise.com/mellow-128",
        "pop": "https://stream.radioparadise.com/main-128",
        "edm": "https://stream.radioparadise.com/dance-128",
    }
    url = radio_urls.get(station.lower(), station)
    if not url.startswith("http"):
        await interaction.followup.send(":x: Unknown station. Try: lofi, jazz, rock, chill, pop, edm, or paste a stream URL.")
        return

    if not voice_client:
        voice_client = await voice.connect()
    elif voice_client.channel != voice:
        await voice_client.move_to(voice)

    try:
        source = discord.PCMVolumeTransformer(
            discord.FFmpegPCMAudio(url, before_options="-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5")
        )
        voice_client.play(source)
        await interaction.followup.send(f":radio: Playing **{station}**")
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}")


@bot.tree.command(name="stopradio", description="Stop radio and leave voice")
async def slash_stopradio(interaction: discord.Interaction):
    vc = interaction.guild.voice_client
    if vc:
        _voice_recv_active.pop(interaction.guild_id, None)
        _voice_text_channels.pop(interaction.guild_id, None)
        _voice_recv_locks.pop(interaction.guild_id, None)
        _voice_tts_lang.pop(interaction.guild_id, None)
        _voice_last_speech.pop(interaction.guild_id, None)
        worker = _tts_queue_workers.pop(interaction.guild_id, None)
        if worker:
            worker.cancel()
        q = _tts_queues.pop(interaction.guild_id, None)
        if q:
            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break
        vc.stop()
        await vc.disconnect()
    await interaction.response.send_message(":stop_button: Stopped.")


@bot.tree.command(name="play", description="Play a song from YouTube or search by name")
@app_commands.describe(query="Song name or YouTube URL")
async def slash_play(interaction: discord.Interaction, query: str):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    if not interaction.user.voice:
        await interaction.response.send_message(":x: Join a voice channel first!", ephemeral=True)
        return
    await interaction.response.defer()

    voice = interaction.user.voice.channel
    player = interaction.guild.voice_client
    if not player:
        try:
            player = await voice.connect(cls=wavelink.Player)
        except Exception as e:
            err_str = str(e).lower()
            if "wavelink" in err_str or "node" in err_str or "pool" in err_str:
                _fallback_nodes = [
                    wavelink.Node(uri="http://lavalink.jirayu.net:13592", password="youshallnotpass", retries=1),
                    wavelink.Node(uri="http://lavalinkv4.serenetia.com:80", password="https://seretia.link/discord", retries=1),
                ]
                for _n in _fallback_nodes:
                    try:
                        await asyncio.wait_for(wavelink.Pool.connect(nodes=[_n], client=bot), timeout=15)
                    except:
                        continue
                player = await voice.connect(cls=wavelink.Player)
            else:
                await interaction.followup.send(f":x: Could not join voice: {e}")
                return
    elif player.channel != voice:
        await player.move_to(voice)

    if not hasattr(player, "home"):
        player.home = interaction.channel
    elif player.home != interaction.channel:
        await interaction.followup.send(f"Player is locked to {player.home.mention}.")
        return

    try:
        tracks: wavelink.Search = await wavelink.Playable.search(query)
        if not tracks:
            await interaction.followup.send(":x: No tracks found.")
            return
        player.autoplay = wavelink.AutoPlayMode.partial
        if isinstance(tracks, wavelink.Playlist):
            added = await player.queue.put_wait(tracks)
            await interaction.followup.send(f":cd: Added playlist `{tracks.name}` ({added} tracks)")
        else:
            track = tracks[0]
            await player.queue.put_wait(track)
            await interaction.followup.send(f":musical_note: Added `{track.title}` by `{track.author}` to the queue.")
        if not player.playing:
            await player.play(player.queue.get(), volume=30)
    except Exception as e:
        await interaction.followup.send(f":x: Play error: {e}")


@bot.tree.command(name="skip", description="Skip the current song")
async def slash_skip(interaction: discord.Interaction):
    player = interaction.guild.voice_client if interaction.guild else None
    if not player or not player.playing:
        await interaction.response.send_message(":x: Nothing is playing.", ephemeral=True)
        return
    await player.skip(force=True)
    await interaction.response.send_message(":track_next: Skipped.")


@bot.tree.command(name="pause", description="Pause the music")
async def slash_pause(interaction: discord.Interaction):
    player = interaction.guild.voice_client if interaction.guild else None
    if not player:
        await interaction.response.send_message(":x: Not in voice.", ephemeral=True)
        return
    await player.pause(True)
    await interaction.response.send_message(":pause_button: Paused.")


@bot.tree.command(name="resume", description="Resume the music")
async def slash_resume(interaction: discord.Interaction):
    player = interaction.guild.voice_client if interaction.guild else None
    if not player:
        await interaction.response.send_message(":x: Not in voice.", ephemeral=True)
        return
    await player.pause(False)
    await interaction.response.send_message(":arrow_forward: Resumed.")


@bot.tree.command(name="stop", description="Stop music and leave voice")
async def slash_stop(interaction: discord.Interaction):
    player = interaction.guild.voice_client if interaction.guild else None
    if not player:
        await interaction.response.send_message(":x: Not in voice.", ephemeral=True)
        return
    await player.disconnect()
    await interaction.response.send_message(":stop_button: Stopped and left.")


@bot.tree.command(name="queue", description="Show the current music queue")
async def slash_queue(interaction: discord.Interaction):
    player = interaction.guild.voice_client if interaction.guild else None
    if not player:
        await interaction.response.send_message(":x: Not in voice.", ephemeral=True)
        return
    lines = []
    if player.current:
        lines.append(f":arrow_forward: Now playing: `{player.current.title}` by `{player.current.author}`")
    if player.queue:
        for i, t in enumerate(player.queue[:20]):
            lines.append(f"{i+1}. `{t.title}` by `{t.author}`")
    else:
        lines.append(":x: Queue is empty.")
    await interaction.response.send_message(f":scroll: **Queue:**\n" + "\n".join(lines))


@bot.tree.command(name="volume", description="Set music volume (0-100)")
@app_commands.describe(level="Volume level")
async def slash_volume(interaction: discord.Interaction, level: int):
    player = interaction.guild.voice_client if interaction.guild else None
    if not player:
        await interaction.response.send_message(":x: Not in voice.", ephemeral=True)
        return
    level = max(0, min(100, level))
    await player.set_volume(level * 10)
    await interaction.response.send_message(f":sound: Volume set to {level}%.")


# =============================================================================
# Temporary Voice Channels (like TempVoice.xyz)
# =============================================================================

def _format_temp_channel_name(fmt: str, member: discord.Member, channel_num: int = 0) -> str:
    name = fmt
    name = name.replace("{OWNER_USERNAME}", member.name)
    name = name.replace("{OWNER_NICKNAME}", member.display_name)
    name = name.replace("{OWNER_MENTION}", member.mention)
    name = name.replace("{OWNER_CREATED}", member.created_at.strftime("%d/%m/%Y") if member.created_at else "?")
    name = name.replace("{OWNER_JOINED}", member.joined_at.strftime("%d/%m/%Y") if member.joined_at else "?")
    name = name.replace("{GUILD_ID}", str(member.guild.id))
    name = name.replace("{NUMBER}", str(channel_num))
    name = name.replace("{NUMBER_ROMAN}", ["", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"][min(channel_num, 10)])
    name = name.replace("{NUMBER_ALPHA}", chr(64 + min(max(channel_num, 1), 26)))
    name = name.replace("{NUMBER_DIGIT}", f"{channel_num:03d}")
    # Clean up excessive whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name[:100]


def _is_creator_channel(channel_id: int) -> bool:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM temp_voice_creators WHERE channel_id = ?", (channel_id,))
    row = cur.fetchone()
    conn.close()
    return row is not None


def _get_creator_config(channel_id: int) -> dict | None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT guild_id, channel_id, name_format, user_limit, privacy_mode, category_id FROM temp_voice_creators WHERE channel_id = ?",
        (channel_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "guild_id": row[0],
        "channel_id": row[1],
        "name_format": row[2],
        "user_limit": row[3],
        "privacy_mode": row[4],
        "category_id": row[5],
    }


def _get_temp_channel(channel_id: int) -> dict | None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT channel_id, guild_id, owner_id, creator_id, created_at, name, user_limit, privacy FROM temp_voice_active WHERE channel_id = ?",
        (channel_id,),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "channel_id": row[0],
        "guild_id": row[1],
        "owner_id": row[2],
        "creator_id": row[3],
        "created_at": row[4],
        "name": row[5],
        "user_limit": row[6],
        "privacy": row[7],
    }


def _save_temp_channel(channel_id: int, guild_id: int, owner_id: int, creator_id: int, **kwargs):
    conn = get_db()
    now = datetime.now(timezone.utc).timestamp()
    conn.execute(
        """INSERT INTO temp_voice_active (channel_id, guild_id, owner_id, creator_id, created_at, name, user_limit, privacy)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(channel_id) DO UPDATE SET
             owner_id=excluded.owner_id, name=COALESCE(excluded.name, temp_voice_active.name),
             user_limit=COALESCE(excluded.user_limit, temp_voice_active.user_limit),
             privacy=COALESCE(excluded.privacy, temp_voice_active.privacy)""",
        (channel_id, guild_id, owner_id, creator_id, now,
         kwargs.get("name"), kwargs.get("user_limit", 0), kwargs.get("privacy", "public")),
    )
    conn.commit()
    conn.close()


def _remove_temp_channel(channel_id: int):
    conn = get_db()
    conn.execute("DELETE FROM temp_voice_active WHERE channel_id = ?", (channel_id,))
    conn.commit()
    conn.close()


def _get_user_prefs(guild_id: int, user_id: int) -> dict:
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT saved_name, saved_limit, saved_privacy FROM temp_voice_prefs WHERE guild_id = ? AND user_id = ?",
        (guild_id, user_id),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"saved_name": None, "saved_limit": None, "saved_privacy": None}
    return {"saved_name": row[0], "saved_limit": row[1], "saved_privacy": row[2]}


def _save_user_prefs(guild_id: int, user_id: int, **kwargs):
    conn = get_db()
    existing = _get_user_prefs(guild_id, user_id)
    name = kwargs.get("saved_name", existing["saved_name"])
    limit = kwargs.get("saved_limit", existing["saved_limit"])
    privacy = kwargs.get("saved_privacy", existing["saved_privacy"])
    conn.execute(
        """INSERT INTO temp_voice_prefs (guild_id, user_id, saved_name, saved_limit, saved_privacy)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(guild_id, user_id) DO UPDATE SET
             saved_name=excluded.saved_name, saved_limit=excluded.saved_limit,
             saved_privacy=excluded.saved_privacy""",
        (guild_id, user_id, name, limit, privacy),
    )
    conn.commit()
    conn.close()


async def _create_temp_channel(member: discord.Member, creator_channel: discord.VoiceChannel):
    guild = member.guild
    config = _get_creator_config(creator_channel.id)
    if not config:
        return

    # Check existing temp channels for this creator to determine number
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM temp_voice_active WHERE guild_id = ? AND creator_id = ?",
        (guild.id, creator_channel.id),
    )
    count = cur.fetchone()[0] + 1
    conn.close()

    prefs = _get_user_prefs(guild.id, member.id)
    name = _format_temp_channel_name(config["name_format"], member, count)

    # Pick category
    category = None
    if config["category_id"]:
        cat = guild.get_channel(int(config["category_id"]))
        if cat and isinstance(cat, discord.CategoryChannel):
            # Check if category has room (Discord limit: 50 ch per cat)
            if len(cat.channels) < 50:
                category = cat

    overwrites = {
        guild.default_role: discord.PermissionOverwrite(connect=True, view_channel=True, speak=True),
        member: discord.PermissionOverwrite(manage_channels=True, move_members=True, mute_members=True, deafen_members=True, connect=True, speak=True),
        guild.me: discord.PermissionOverwrite(manage_channels=True, manage_permissions=True, connect=True, speak=True, view_channel=True),
    }

    try:
        temp_channel = await guild.create_voice_channel(
            name=name,
            category=category,
            overwrites=overwrites,
            reason=f"TempVoice: {member} joined creator #{creator_channel.name}",
        )
    except Exception as e:
        print(f"[TEMP VOICE] Failed to create channel: {e}")
        return

    _save_temp_channel(
        temp_channel.id, guild.id, member.id, creator_channel.id,
        name=name, user_limit=config["user_limit"], privacy=config["privacy_mode"],
    )

    if config["user_limit"] > 0:
        await temp_channel.edit(user_limit=config["user_limit"])

    try:
        await member.move_to(temp_channel, reason="TempVoice: Moved to new channel")
    except Exception:
        pass

    print(f"[TEMP VOICE] Created {temp_channel.name} ({temp_channel.id}) for {member}")
    await _send_temp_interface(temp_channel, member)


async def _delete_temp_channel(channel: discord.VoiceChannel):
    if not _get_temp_channel(channel.id):
        return
    try:
        await channel.delete(reason="TempVoice: Channel empty")
        _remove_temp_channel(channel.id)
        print(f"[TEMP VOICE] Deleted temp channel {channel.name} ({channel.id})")
    except Exception as e:
        print(f"[TEMP VOICE] Failed to delete {channel.name}: {e}")


async def _handle_owner_left(channel: discord.VoiceChannel, current_members: list[discord.Member]):
    record = _get_temp_channel(channel.id)
    if not record:
        return
    # If no one is in the channel, delete it
    if len(current_members) == 0:
        await _delete_temp_channel(channel)
        return
    # Transfer ownership to the next person who's been there longest
    new_owner = current_members[0]
    _save_temp_channel(channel.id, channel.guild.id, new_owner.id, record["creator_id"],
                       name=record["name"], user_limit=record["user_limit"], privacy=record["privacy"])
    print(f"[TEMP VOICE] Ownership transferred to {new_owner} for {channel.name}")


async def _reapply_privacy(channel: discord.VoiceChannel, privacy: str, owner_id: int, trusted: set[int], blocked: set[int]):
    guild = channel.guild
    overwrites = {
        guild.default_role: discord.PermissionOverwrite(),
        guild.me: discord.PermissionOverwrite(manage_channels=True, manage_permissions=True, connect=True, speak=True, view_channel=True),
    }
    owner = guild.get_member(owner_id)
    if owner:
        overwrites[owner] = discord.PermissionOverwrite(manage_channels=True, move_members=True, mute_members=True, deafen_members=True, connect=True, speak=True)

    if privacy == "public":
        overwrites[guild.default_role].connect = True
        overwrites[guild.default_role].view_channel = True
        overwrites[guild.default_role].speak = True
    elif privacy == "locked":
        overwrites[guild.default_role].connect = False
        overwrites[guild.default_role].view_channel = True
        overwrites[guild.default_role].speak = True
    elif privacy == "hidden":
        overwrites[guild.default_role].connect = False
        overwrites[guild.default_role].view_channel = False

    for uid in trusted:
        u = guild.get_member(uid)
        if u:
            overwrites[u] = discord.PermissionOverwrite(connect=True, view_channel=True, speak=True)

    for uid in blocked:
        u = guild.get_member(uid)
        if u:
            overwrites[u] = discord.PermissionOverwrite(connect=False, view_channel=False)

    try:
        await channel.edit(overwrites=overwrites)
    except Exception as e:
        print(f"[TEMP VOICE] Failed to reapply privacy: {e}")


# --- Voice State Event ---

@bot.event
async def on_voice_state_update(member: discord.Member, before: discord.VoiceState, after: discord.VoiceState):
    if member.bot:
        return

    # User left a channel
    if before.channel and before.channel != after.channel:
        old_channel = before.channel
        if _is_creator_channel(old_channel.id):
            pass  # They left a creator - nothing to do
        elif _get_temp_channel(old_channel.id):
            remaining = [m for m in old_channel.members if not m.bot]
            # Check if owner left
            record = _get_temp_channel(old_channel.id)
            if record and record["owner_id"] == member.id:
                await _handle_owner_left(old_channel, remaining)
            elif len(remaining) == 0:
                await _delete_temp_channel(old_channel)

    # User joined a channel
    if after.channel and after.channel != before.channel:
        new_channel = after.channel
        if _is_creator_channel(new_channel.id):
            # Check if they already own a temp channel
            guild_id = member.guild.id
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "SELECT channel_id FROM temp_voice_active WHERE guild_id = ? AND owner_id = ?",
                (guild_id, member.id),
            )
            existing = cur.fetchone()
            conn.close()
            if existing:
                # They already have a temp channel - move them to it
                existing_ch = member.guild.get_channel(int(existing[0]))
                if existing_ch:
                    try:
                        await member.move_to(existing_ch, reason="TempVoice: Moving to your existing channel")
                    except Exception:
                        pass
                    return
            # Create a new temp channel
            await _create_temp_channel(member, new_channel)

    # Check if a temp channel needs cleanup (user disconnected)
    if before.channel and not after.channel:
        old_channel = before.channel
        if _get_temp_channel(old_channel.id) and len(old_channel.members) == 0:
            await _delete_temp_channel(old_channel)


# --- TempVoice Admin Commands ---

@app_commands.guild_only()
class TempVoiceAdmin(app_commands.Group):
    pass

tempvoice_group = TempVoiceAdmin(name="tempvoice", description="Manage temporary voice channels")


@tempvoice_group.command(name="setup", description="Mark this voice channel as a creator channel (Admin only)")
@app_commands.describe(channel="Voice channel to set as creator")
async def slash_tv_setup(interaction: discord.Interaction, channel: discord.VoiceChannel):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return

    if _is_creator_channel(channel.id):
        await interaction.response.send_message(f":x: **{channel.name}** is already a creator channel.", ephemeral=True)
        return

    conn = get_db()
    try:
        conn.execute(
            "INSERT INTO temp_voice_creators (guild_id, channel_id) VALUES (?, ?)",
            (interaction.guild.id, channel.id),
        )
        conn.commit()
        await interaction.response.send_message(f":white_check_mark: **{channel.name}** is now a creator channel! Join it to spawn a temp voice channel.", ephemeral=False)
    except Exception as e:
        print(f"[TEMP VOICE] setup error for {channel.id}: {e}")
        await interaction.response.send_message(f":x: Failed to set **{channel.name}** as creator: `{e}`", ephemeral=True)
    finally:
        conn.close()


@tempvoice_group.command(name="remove", description="Remove creator channel status (Admin only)")
@app_commands.describe(channel="Voice channel to remove")
async def slash_tv_remove(interaction: discord.Interaction, channel: discord.VoiceChannel):
    if not is_admin(interaction.user):
        await interaction.response.send_message(":x: Admin only.", ephemeral=True)
        return

    if not _is_creator_channel(channel.id):
        await interaction.response.send_message(f":x: **{channel.name}** is not a creator channel.", ephemeral=True)
        return

    conn = get_db()
    conn.execute("DELETE FROM temp_voice_creators WHERE channel_id = ?", (channel.id,))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: **{channel.name}** is no longer a creator channel.", ephemeral=False)


@tempvoice_group.command(name="config", description="Show current server's creator channels")
async def slash_tv_config(interaction: discord.Interaction):
    await interaction.response.defer()
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "SELECT channel_id, name_format, user_limit, privacy_mode FROM temp_voice_creators WHERE guild_id = ?",
        (interaction.guild.id,),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        await interaction.followup.send(":x: No creator channels set up. Use `/tempvoice setup` on a voice channel.")
        return

    lines = []
    for row in rows:
        ch = interaction.guild.get_channel(int(row[0]))
        ch_name = ch.mention if ch else f"`{row[0]}`"
        lines.append(f"• {ch_name} — Format: `{row[1]}` | Limit: {row[2]} | Privacy: {row[3]}")

    await interaction.followup.send(f":loud_sound: **Creator Channels:**\n" + "\n".join(lines))


# --- /voice Command Group (User Temp Channel Controls) ---

@app_commands.guild_only()
class VoiceControls(app_commands.Group):
    pass

voice_group = VoiceControls(name="voice", description="Manage your temporary voice channel")


def _require_temp_owner(interaction: discord.Interaction):
    """Check that the user owns a temp channel and returns (record, channel) or sends error."""
    record = _get_temp_channel(interaction.channel_id)
    if not record or record["owner_id"] != interaction.user.id:
        # Check if they own any temp channel
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "SELECT channel_id FROM temp_voice_active WHERE guild_id = ? AND owner_id = ?",
            (interaction.guild.id, interaction.user.id),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None, None, ":x: You don't own a temporary voice channel. Join a creator channel first."
        ch = interaction.guild.get_channel(int(row[0]))
        if ch:
            return _get_temp_channel(ch.id), ch, None
        return None, None, ":x: Could not find your temp channel."
    ch = interaction.guild.get_channel(int(record["channel_id"]))
    return record, ch, None


@voice_group.command(name="name", description="Change your temp channel name")
@app_commands.describe(name="New name for your channel")
async def slash_voice_name(interaction: discord.Interaction, name: str):
    await interaction.response.defer(ephemeral=True)
    record, ch, err = _require_temp_owner(interaction)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return
    if len(name) > 100:
        await interaction.followup.send(":x: Name too long (max 100 chars).", ephemeral=True)
        return
    try:
        await ch.edit(name=name)
        _save_temp_channel(ch.id, ch.guild.id, record["owner_id"], record["creator_id"], name=name)
        _save_user_prefs(interaction.guild.id, interaction.user.id, saved_name=name)
        await interaction.followup.send(f":white_check_mark: Channel renamed to **{name}**", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f":x: Failed to rename: {e}", ephemeral=True)


@voice_group.command(name="limit", description="Set user limit for your temp channel (0 = unlimited)")
@app_commands.describe(limit="Max users (0 = no limit)")
async def slash_voice_limit(interaction: discord.Interaction, limit: int):
    await interaction.response.defer(ephemeral=True)
    record, ch, err = _require_temp_owner(interaction)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return
    if limit < 0 or limit > 99:
        await interaction.followup.send(":x: Limit must be between 0 and 99.", ephemeral=True)
        return
    try:
        await ch.edit(user_limit=limit)
        _save_temp_channel(ch.id, ch.guild.id, record["owner_id"], record["creator_id"], user_limit=limit)
        _save_user_prefs(interaction.guild.id, interaction.user.id, saved_limit=limit)
        msg = f":white_check_mark: User limit set to **{limit}**." if limit > 0 else ":white_check_mark: User limit removed (unlimited)."
        await interaction.followup.send(msg, ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


@voice_group.command(name="privacy", description="Set your channel privacy")
@app_commands.describe(mode="public (open), locked (visible but no join), hidden (invisible)")
async def slash_voice_privacy(interaction: discord.Interaction, mode: str):
    await interaction.response.defer(ephemeral=True)
    record, ch, err = _require_temp_owner(interaction)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return
    mode = mode.lower()
    if mode not in ("public", "locked", "hidden"):
        await interaction.followup.send(":x: Mode must be: public, locked, or hidden.", ephemeral=True)
        return
    try:
        guild = ch.guild
        overwrites = ch.overwrites
        default_perms = overwrites.get(guild.default_role, discord.PermissionOverwrite())
        if mode == "public":
            default_perms.connect = True
            default_perms.view_channel = True
            default_perms.speak = True
        elif mode == "locked":
            default_perms.connect = False
            default_perms.view_channel = True
            default_perms.speak = True
        elif mode == "hidden":
            default_perms.connect = False
            default_perms.view_channel = False
        overwrites[guild.default_role] = default_perms
        await ch.edit(overwrites=overwrites)
        _save_temp_channel(ch.id, ch.guild.id, record["owner_id"], record["creator_id"], privacy=mode)
        _save_user_prefs(interaction.guild.id, interaction.user.id, saved_privacy=mode)
        icon = {"public": "🔓", "locked": "🔒", "hidden": "😎"}
        await interaction.followup.send(f"{icon.get(mode, '')} Privacy set to **{mode}**.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


@voice_group.command(name="trust", description="Allow a user to join your locked/hidden channel")
@app_commands.describe(user="User to trust")
async def slash_voice_trust(interaction: discord.Interaction, user: discord.Member):
    await interaction.response.defer(ephemeral=True)
    record, ch, err = _require_temp_owner(interaction)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return
    try:
        await ch.set_permissions(user, connect=True, view_channel=True, speak=True)
        await interaction.followup.send(f":white_check_mark: {user.mention} can now join your channel.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


@voice_group.command(name="untrust", description="Remove a user's special access")
@app_commands.describe(user="User to untrust")
async def slash_voice_untrust(interaction: discord.Interaction, user: discord.Member):
    await interaction.response.defer(ephemeral=True)
    record, ch, err = _require_temp_owner(interaction)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return
    try:
        await ch.set_permissions(user, overwrite=None)
        await interaction.followup.send(f":white_check_mark: {user.mention}'s access removed.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


@voice_group.command(name="block", description="Block a user from joining")
@app_commands.describe(user="User to block")
async def slash_voice_block(interaction: discord.Interaction, user: discord.Member):
    await interaction.response.defer(ephemeral=True)
    record, ch, err = _require_temp_owner(interaction)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return
    try:
        await ch.set_permissions(user, connect=False, view_channel=False)
        if user in ch.members:
            await user.move_to(None, reason="TempVoice: Blocked by owner")
        await interaction.followup.send(f":white_check_mark: {user.mention} blocked.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


@voice_group.command(name="unblock", description="Unblock a user")
@app_commands.describe(user="User to unblock")
async def slash_voice_unblock(interaction: discord.Interaction, user: discord.Member):
    await interaction.response.defer(ephemeral=True)
    record, ch, err = _require_temp_owner(interaction)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return
    try:
        await ch.set_permissions(user, overwrite=None)
        await interaction.followup.send(f":white_check_mark: {user.mention} unblocked.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


@voice_group.command(name="kick", description="Kick a user from your temp channel")
@app_commands.describe(user="User to kick")
async def slash_voice_kick(interaction: discord.Interaction, user: discord.Member):
    await interaction.response.defer(ephemeral=True)
    record, ch, err = _require_temp_owner(interaction)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return
    if user not in ch.members:
        await interaction.followup.send(f":x: {user.mention} is not in your channel.", ephemeral=True)
        return
    try:
        await user.move_to(None, reason=f"TempVoice: Kicked by {interaction.user}")
        # Lock the channel briefly so they can't rejoin
        await ch.set_permissions(user, connect=False)
        await interaction.followup.send(f":boot: {user.mention} kicked from your channel.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


@voice_group.command(name="invite", description="Invite a user to your temp channel")
@app_commands.describe(user="User to invite")
async def slash_voice_invite(interaction: discord.Interaction, user: discord.Member):
    await interaction.response.defer(ephemeral=True)
    record, ch, err = _require_temp_owner(interaction)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return
    try:
        await ch.set_permissions(user, connect=True, view_channel=True, speak=True)
        await interaction.followup.send(f":envelope: Invited {user.mention} to **{ch.name}**", ephemeral=True)
        try:
            await user.send(f":wave: {interaction.user.display_name} invited you to **{ch.name}** in **{ch.guild.name}**!\nJoin: <#{ch.id}>")
        except Exception:
            pass
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


@voice_group.command(name="claim", description="Claim ownership of a temp channel (if owner left)")
async def slash_voice_claim(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    # The user must be IN a temp channel that has an absent owner
    if not interaction.user.voice or not interaction.user.voice.channel:
        await interaction.followup.send(":x: You're not in a voice channel.", ephemeral=True)
        return
    ch = interaction.user.voice.channel
    record = _get_temp_channel(ch.id)
    if not record:
        await interaction.followup.send(":x: This is not a temporary voice channel.", ephemeral=True)
        return
    owner = ch.guild.get_member(int(record["owner_id"]))
    if owner and owner in ch.members:
        await interaction.followup.send(":x: The owner is still in the channel.", ephemeral=True)
        return
    # Transfer ownership
    _save_temp_channel(ch.id, ch.guild.id, interaction.user.id, record["creator_id"],
                       name=record["name"], user_limit=record["user_limit"], privacy=record["privacy"])
    await ch.set_permissions(interaction.user, manage_channels=True, move_members=True, mute_members=True, deafen_members=True, connect=True, speak=True)
    if owner:
        try:
            await ch.set_permissions(owner, overwrite=None)
        except Exception:
            pass
    await interaction.followup.send(f":white_check_mark: You are now the owner of **{ch.name}**!", ephemeral=True)


@voice_group.command(name="transfer", description="Transfer ownership to another user in your channel")
@app_commands.describe(user="New owner")
async def slash_voice_transfer(interaction: discord.Interaction, user: discord.Member):
    await interaction.response.defer(ephemeral=True)
    record, ch, err = _require_temp_owner(interaction)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return
    if user == interaction.user:
        await interaction.followup.send(":x: You already own the channel.", ephemeral=True)
        return
    if user not in ch.members:
        await interaction.followup.send(f":x: {user.mention} is not in your channel.", ephemeral=True)
        return
    try:
        await ch.set_permissions(interaction.user, overwrite=None)
        await ch.set_permissions(user, manage_channels=True, move_members=True, mute_members=True, deafen_members=True, connect=True, speak=True)
        _save_temp_channel(ch.id, ch.guild.id, user.id, record["creator_id"],
                           name=record["name"], user_limit=record["user_limit"], privacy=record["privacy"])
        await interaction.followup.send(f":white_check_mark: Ownership transferred to {user.mention}.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


@voice_group.command(name="info", description="Show info about your temp channel")
async def slash_voice_info(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    record, ch, err = _require_temp_owner(interaction)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return
    owner = ch.guild.get_member(int(record["owner_id"]))
    owner_name = owner.mention if owner else f"User {record['owner_id']}"
    created = datetime.fromtimestamp(record["created_at"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if record["created_at"] else "?"
    embed = discord.Embed(title=f":loud_sound: {ch.name}", color=discord.Color.blue())
    embed.add_field(name="Owner", value=owner_name, inline=True)
    embed.add_field(name="Users", value=f"{len(ch.members)}/{ch.user_limit or '∞'}", inline=True)
    embed.add_field(name="Privacy", value=record["privacy"].title(), inline=True)
    embed.add_field(name="Created", value=created, inline=False)
    await interaction.followup.send(embed=embed, ephemeral=True)


@voice_group.command(name="reset", description="Reset your temp channel to defaults")
async def slash_voice_reset(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    record, ch, err = _require_temp_owner(interaction)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return
    config = _get_creator_config(record["creator_id"])
    default_name = ch.name
    if config:
        default_name = _format_temp_channel_name(config["name_format"], interaction.user, 0)
    try:
        guild = ch.guild
        overwrites = {
            guild.default_role: discord.PermissionOverwrite(connect=True, view_channel=True, speak=True),
            guild.me: discord.PermissionOverwrite(manage_channels=True, manage_permissions=True, connect=True, speak=True, view_channel=True),
        }
        await ch.edit(name=default_name, user_limit=0, overwrites=overwrites)
        _save_temp_channel(ch.id, ch.guild.id, record["owner_id"], record["creator_id"],
                           name=default_name, user_limit=0, privacy="public")
        await interaction.followup.send(f":white_check_mark: Channel reset to defaults: **{default_name}**", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


@voice_group.command(name="delete", description="Force delete your temp channel")
async def slash_voice_delete(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    record, ch, err = _require_temp_owner(interaction)
    if err:
        await interaction.followup.send(err, ephemeral=True)
        return
    try:
        _remove_temp_channel(ch.id)
        await ch.delete(reason=f"TempVoice: Deleted by owner {interaction.user}")
        await interaction.followup.send(":white_check_mark: Channel deleted.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


# --- Voice Interface View (TempVoice-style button controls) ---

PRIVACY_CYCLE = ["public", "locked", "hidden"]
PRIVACY_LABELS = {"public": "🔓 Public", "locked": "🔒 Locked", "hidden": "😎 Hidden"}
PRIVACY_EMOJIS = {"public": "🔓", "locked": "🔒", "hidden": "😎"}


async def _send_temp_interface(target_channel: discord.VoiceChannel, owner: discord.Member):
    """Send the control interface to the voice channel's chat."""
    record = _get_temp_channel(target_channel.id)
    if not record:
        return
    try:
        embed = discord.Embed(
            title=f"🎧 {target_channel.name} — Controls",
            description=f"Owner: {owner.mention}\nUse the buttons below to manage your channel.",
            color=0x2b2d31,
        )
        embed.set_footer(text="TempVoice • You're the owner")
        view = TempVoiceInterface(target_channel.id, record.get("privacy", "public"))
        await target_channel.send(embed=embed, view=view)
        print(f"[TEMP VOICE] Sent control interface for {target_channel.name}")
    except Exception as e:
        print(f"[TEMP VOICE] Failed to send interface: {e}")


def _toggle_privacy(current: str) -> str:
    idx = PRIVACY_CYCLE.index(current) if current in PRIVACY_CYCLE else 0
    return PRIVACY_CYCLE[(idx + 1) % len(PRIVACY_CYCLE)]


async def _apply_privacy(ch: discord.VoiceChannel, mode: str):
    guild = ch.guild
    overwrites = ch.overwrites
    default_perms = overwrites.get(guild.default_role, discord.PermissionOverwrite())
    if mode == "public":
        default_perms.connect = True
        default_perms.view_channel = True
        default_perms.speak = True
    elif mode == "locked":
        default_perms.connect = False
        default_perms.view_channel = True
        default_perms.speak = True
    else:  # hidden
        default_perms.connect = False
        default_perms.view_channel = False
    overwrites[guild.default_role] = default_perms
    # Preserve owner override
    overwrites[guild.me] = discord.PermissionOverwrite(manage_channels=True, manage_permissions=True, connect=True, speak=True, view_channel=True)
    await ch.edit(overwrites=overwrites)


def _check_owner(interaction: discord.Interaction, target_channel_id: int) -> tuple[dict | None, discord.VoiceChannel | None, str | None]:
    ch = interaction.guild.get_channel(target_channel_id)
    if not ch or not isinstance(ch, discord.VoiceChannel):
        return None, None, ":x: Channel not found."
    record = _get_temp_channel(ch.id)
    if not record:
        return None, None, ":x: Not a temporary channel."
    if record["owner_id"] != interaction.user.id:
        return None, None, ":x: You don't own this channel."
    return record, ch, None


class TempVoiceInterface(discord.ui.View):
    def __init__(self, target_channel_id: int, current_privacy: str = "public"):
        super().__init__(timeout=None)
        self.target_channel_id = target_channel_id
        self.current_privacy = current_privacy

    async def on_error(self, interaction: discord.Interaction, error: Exception, item: discord.ui.Item):
        print(f"[TEMP VOICE] Button {item.custom_id} error: {error}")
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(":x: Something went wrong. Try again.", ephemeral=True)
        except Exception:
            pass

    # Row 1: Name, Limit, Privacy toggle
    @discord.ui.button(label="Name", emoji="✏️", style=discord.ButtonStyle.secondary, custom_id="tvif_name")
    async def tv_name(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.send_modal(TempVoiceNameModal(self.target_channel_id))
        except Exception as e:
            print(f"[TEMP VOICE] tv_name error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(":x: Failed to open name editor.", ephemeral=True)

    @discord.ui.button(label="Limit", emoji="🔢", style=discord.ButtonStyle.secondary, custom_id="tvif_limit")
    async def tv_limit(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.send_modal(TempVoiceLimitModal(self.target_channel_id))
        except Exception as e:
            print(f"[TEMP VOICE] tv_limit error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(":x: Failed to open limit editor.", ephemeral=True)

    @discord.ui.button(label="Public", emoji="🔓", style=discord.ButtonStyle.success, custom_id="tvif_privacy")
    async def tv_privacy(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.defer(ephemeral=True)
            record, ch, err = _check_owner(interaction, self.target_channel_id)
            if err:
                await interaction.followup.send(err, ephemeral=True)
                return
            new_mode = _toggle_privacy(record["privacy"])
            await _apply_privacy(ch, new_mode)
            _save_temp_channel(ch.id, ch.guild.id, record["owner_id"], record["creator_id"], privacy=new_mode)
            _save_user_prefs(interaction.guild.id, interaction.user.id, saved_privacy=new_mode)
            # Update the button label
            button.label = PRIVACY_LABELS.get(new_mode, "Public")
            button.emoji = PRIVACY_EMOJIS.get(new_mode, "🔓")
            style_map = {"public": discord.ButtonStyle.success, "locked": discord.ButtonStyle.primary, "hidden": discord.ButtonStyle.danger}
            button.style = style_map.get(new_mode, discord.ButtonStyle.success)
            self.current_privacy = new_mode
            await interaction.followup.edit_message(interaction.message.id, view=self)
        except Exception as e:
            print(f"[TEMP VOICE] tv_privacy error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(f":x: {e}", ephemeral=True)
            else:
                await interaction.followup.send(f":x: {e}", ephemeral=True)

    # Row 2: Waiting Room, Chat Thread
    @discord.ui.button(label="Waiting", emoji="🕐", style=discord.ButtonStyle.secondary, custom_id="tvif_waiting")
    async def tv_waiting(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.defer(ephemeral=True)
            record, ch, err = _check_owner(interaction, self.target_channel_id)
            if err:
                await interaction.followup.send(err, ephemeral=True)
                return
            guild = ch.guild
            existing = discord.utils.get(guild.voice_channels, name=f"waiting-{ch.id}")
            if existing:
                await interaction.followup.send(":x: Waiting room already exists.", ephemeral=True)
                return
            wait_ch = await guild.create_voice_channel(
                name=f"⏳ {ch.name} Waiting",
                category=ch.category,
                position=ch.position + 1,
                user_limit=0,
            )
            await ch.set_permissions(guild.default_role, connect=False, view_channel=False)
            await interaction.followup.send(f":white_check_mark: Waiting room created: {wait_ch.mention}\nUsers must be trusted to enter your main channel.", ephemeral=True)
        except Exception as e:
            print(f"[TEMP VOICE] tv_waiting error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(f":x: {e}", ephemeral=True)
            else:
                await interaction.followup.send(f":x: {e}", ephemeral=True)

    @discord.ui.button(label="Chat Thread", emoji="💬", style=discord.ButtonStyle.secondary, custom_id="tvif_thread")
    async def tv_thread(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.defer(ephemeral=True)
            record, ch, err = _check_owner(interaction, self.target_channel_id)
            if err:
                await interaction.followup.send(err, ephemeral=True)
                return
            thread = await ch.create_thread(name=f"💬 {ch.name} Chat", type=discord.ChannelType.public_thread)
            await interaction.followup.send(f":white_check_mark: Thread created: {thread.mention}", ephemeral=True)
        except Exception as e:
            print(f"[TEMP VOICE] tv_thread error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(f":x: {e}", ephemeral=True)
            else:
                await interaction.followup.send(f":x: {e}", ephemeral=True)

    # Row 3: Trust, Untrust, Block, Unblock
    @discord.ui.button(label="Trust", emoji="✅", style=discord.ButtonStyle.success, custom_id="tvif_trust")
    async def tv_trust(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.send_modal(TempVoiceUserModal(self.target_channel_id, "trust"))
        except Exception as e:
            print(f"[TEMP VOICE] tv_trust error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(":x: Failed to open user editor.", ephemeral=True)

    @discord.ui.button(label="Untrust", emoji="➖", style=discord.ButtonStyle.secondary, custom_id="tvif_untrust")
    async def tv_untrust(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.send_modal(TempVoiceUserModal(self.target_channel_id, "untrust"))
        except Exception as e:
            print(f"[TEMP VOICE] tv_untrust error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(":x: Failed to open user editor.", ephemeral=True)

    @discord.ui.button(label="Block", emoji="🚫", style=discord.ButtonStyle.danger, custom_id="tvif_block")
    async def tv_block(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.send_modal(TempVoiceUserModal(self.target_channel_id, "block"))
        except Exception as e:
            print(f"[TEMP VOICE] tv_block error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(":x: Failed to open user editor.", ephemeral=True)

    @discord.ui.button(label="Unblock", emoji="♻️", style=discord.ButtonStyle.secondary, custom_id="tvif_unblock")
    async def tv_unblock(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.send_modal(TempVoiceUserModal(self.target_channel_id, "unblock"))
        except Exception as e:
            print(f"[TEMP VOICE] tv_unblock error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(":x: Failed to open user editor.", ephemeral=True)

    # Row 4: Invite, Kick, Info, Reset, Delete
    @discord.ui.button(label="Invite", emoji="📨", style=discord.ButtonStyle.secondary, custom_id="tvif_invite")
    async def tv_invite(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.send_modal(TempVoiceUserModal(self.target_channel_id, "invite"))
        except Exception as e:
            print(f"[TEMP VOICE] tv_invite error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(":x: Failed to open user editor.", ephemeral=True)

    @discord.ui.button(label="Kick", emoji="👢", style=discord.ButtonStyle.danger, custom_id="tvif_kick")
    async def tv_kick(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.send_modal(TempVoiceUserModal(self.target_channel_id, "kick"))
        except Exception as e:
            print(f"[TEMP VOICE] tv_kick error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(":x: Failed to open user editor.", ephemeral=True)

    @discord.ui.button(label="Info", emoji="📋", style=discord.ButtonStyle.secondary, custom_id="tvif_info")
    async def tv_info(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.defer(ephemeral=True)
            record, ch, err = _check_owner(interaction, self.target_channel_id)
            if err:
                await interaction.followup.send(err, ephemeral=True)
                return
            owner = ch.guild.get_member(int(record["owner_id"]))
            owner_name = owner.mention if owner else f"User {record['owner_id']}"
            created = datetime.fromtimestamp(record["created_at"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if record["created_at"] else "?"
            embed = discord.Embed(title=f"🎧 {ch.name}", color=0x2b2d31)
            embed.add_field(name="Owner", value=owner_name, inline=True)
            embed.add_field(name="Users", value=f"{len(ch.members)}/{ch.user_limit or '∞'}", inline=True)
            embed.add_field(name="Privacy", value=record["privacy"].title(), inline=True)
            embed.add_field(name="Created", value=created, inline=False)
            await interaction.followup.send(embed=embed, ephemeral=True)
        except Exception as e:
            print(f"[TEMP VOICE] tv_info error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(f":x: {e}", ephemeral=True)
            else:
                await interaction.followup.send(f":x: {e}", ephemeral=True)

    @discord.ui.button(label="Reset", emoji="🔄", style=discord.ButtonStyle.secondary, custom_id="tvif_reset")
    async def tv_reset(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.defer(ephemeral=True)
            record, ch, err = _check_owner(interaction, self.target_channel_id)
            if err:
                await interaction.followup.send(err, ephemeral=True)
                return
            config = _get_creator_config(record["creator_id"])
            default_name = config["name_format"] if config else f"{interaction.user.display_name}s Channel"
            default_name = _format_temp_channel_name(default_name, interaction.user, 0)
            await ch.edit(name=default_name, user_limit=0)
            await _apply_privacy(ch, "public")
            _save_temp_channel(ch.id, ch.guild.id, record["owner_id"], record["creator_id"],
                               name=default_name, user_limit=0, privacy="public")
            await interaction.followup.send(":white_check_mark: Reset to defaults.", ephemeral=True)
        except Exception as e:
            print(f"[TEMP VOICE] tv_reset error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(f":x: {e}", ephemeral=True)
            else:
                await interaction.followup.send(f":x: {e}", ephemeral=True)

    @discord.ui.button(label="Delete", emoji="🗑️", style=discord.ButtonStyle.danger, custom_id="tvif_delete")
    async def tv_delete(self, interaction: discord.Interaction, button: discord.ui.Button):
        try:
            await interaction.response.defer(ephemeral=True)
            record, ch, err = _check_owner(interaction, self.target_channel_id)
            if err:
                await interaction.followup.send(err, ephemeral=True)
                return
            _remove_temp_channel(ch.id)
            await ch.delete(reason=f"TempVoice: Deleted by owner {interaction.user}")
            await interaction.followup.send(":white_check_mark: Channel deleted.", ephemeral=True)
        except Exception as e:
            print(f"[TEMP VOICE] tv_delete error: {e}")
            if not interaction.response.is_done():
                await interaction.response.send_message(f":x: {e}", ephemeral=True)
            else:
                await interaction.followup.send(f":x: {e}", ephemeral=True)


# Modals for the interface

class TempVoiceNameModal(discord.ui.Modal, title="Change Channel Name"):
    name_input = discord.ui.TextInput(label="New name", max_length=100)

    def __init__(self, target_channel_id: int):
        super().__init__()
        self.target_channel_id = target_channel_id

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        record, ch, err = _check_owner(interaction, self.target_channel_id)
        if err:
            await interaction.followup.send(err, ephemeral=True)
            return
        try:
            name = self.name_input.value.strip()[:100]
            await ch.edit(name=name)
            _save_temp_channel(ch.id, ch.guild.id, record["owner_id"], record["creator_id"], name=name)
            _save_user_prefs(interaction.guild.id, interaction.user.id, saved_name=name)
            await interaction.followup.send(f":white_check_mark: Renamed to **{name}**", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


class TempVoiceLimitModal(discord.ui.Modal, title="Set User Limit"):
    limit_input = discord.ui.TextInput(label="User limit (0 = unlimited)", max_length=2, default="0")

    def __init__(self, target_channel_id: int):
        super().__init__()
        self.target_channel_id = target_channel_id

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer()
        record, ch, err = _check_owner(interaction, self.target_channel_id)
        if err:
            await interaction.followup.send(err, ephemeral=True)
            return
        try:
            limit = max(0, min(99, int(self.limit_input.value.strip())))
            await ch.edit(user_limit=limit)
            _save_temp_channel(ch.id, ch.guild.id, record["owner_id"], record["creator_id"], user_limit=limit)
            _save_user_prefs(interaction.guild.id, interaction.user.id, saved_limit=limit)
            msg = f":white_check_mark: Limit set to **{limit}**." if limit > 0 else ":white_check_mark: Limit removed."
            await interaction.followup.send(msg, ephemeral=True)
        except ValueError:
            await interaction.followup.send(":x: Enter a valid number (0-99).", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


class TempVoiceUserModal(discord.ui.Modal, title="Manage User"):
    user_input = discord.ui.TextInput(label="User ID or mention", placeholder="Paste user ID or @mention")
    def __init__(self, target_channel_id: int, action: str):
        super().__init__(title=f"{action.title()} User")
        self.target_channel_id = target_channel_id
        self.action = action

    async def on_submit(self, interaction: discord.Interaction):
        await interaction.response.defer(ephemeral=True)
        record, ch, err = _check_owner(interaction, self.target_channel_id)
        if err:
            await interaction.followup.send(err, ephemeral=True)
            return
        raw = self.user_input.value.strip()
        uid = re.sub(r"[<@!>]", "", raw)
        if not uid.isdigit():
            await interaction.followup.send(":x: Provide a valid user ID or @mention.", ephemeral=True)
            return
        user = interaction.guild.get_member(int(uid))
        if not user:
            await interaction.followup.send(":x: User not found on this server.", ephemeral=True)
            return

        try:
            if self.action == "trust":
                await ch.set_permissions(user, connect=True, view_channel=True, speak=True)
                await interaction.followup.send(f":white_check_mark: {user.mention} trusted.", ephemeral=True)
            elif self.action == "untrust":
                await ch.set_permissions(user, overwrite=None)
                await interaction.followup.send(f":white_check_mark: {user.mention} untrusted.", ephemeral=True)
            elif self.action == "block":
                await ch.set_permissions(user, connect=False, view_channel=False)
                if user in ch.members:
                    await user.move_to(None, reason="TempVoice: Blocked by owner")
                await interaction.followup.send(f":white_check_mark: {user.mention} blocked.", ephemeral=True)
            elif self.action == "unblock":
                await ch.set_permissions(user, overwrite=None)
                await interaction.followup.send(f":white_check_mark: {user.mention} unblocked.", ephemeral=True)
            elif self.action == "invite":
                await ch.set_permissions(user, connect=True, view_channel=True, speak=True)
                await interaction.followup.send(f":envelope: Invited {user.mention} to **{ch.name}**", ephemeral=True)
                try:
                    await user.send(f":wave: {interaction.user.display_name} invited you to **{ch.name}** in **{ch.guild.name}**!\nJoin: <#{ch.id}>")
                except Exception:
                    pass
            elif self.action == "kick":
                if user in ch.members:
                    await user.move_to(None, reason=f"TempVoice: Kicked by {interaction.user}")
                    await ch.set_permissions(user, connect=False)
                    await interaction.followup.send(f":boot: {user.mention} kicked.", ephemeral=True)
                else:
                    await interaction.followup.send(f":x: {user.mention} is not in your channel.", ephemeral=True)
        except Exception as e:
            await interaction.followup.send(f":x: Failed: {e}", ephemeral=True)


@bot.tree.command(name="tempcontrol", description="Send the temp voice control panel here")
@app_commands.describe(channel="Your temp voice channel (default: channel you're in)")
async def slash_tempcontrol(interaction: discord.Interaction, channel: discord.VoiceChannel = None):
    target = channel or (interaction.user.voice.channel if interaction.user.voice else None)
    if not target:
        await interaction.response.send_message(":x: You're not in a voice channel. Specify one.", ephemeral=True)
        return
    record = _get_temp_channel(target.id)
    if not record or record["owner_id"] != interaction.user.id:
        await interaction.response.send_message(":x: You don't own that channel.", ephemeral=True)
        return
    embed = discord.Embed(
        title=f"🎧 {target.name} — Controls",
        description=f"Owner: {interaction.user.mention}",
        color=0x2b2d31,
    )
    embed.set_footer(text="TempVoice • You're the owner")
    view = TempVoiceInterface(target.id, record.get("privacy", "public"))
    await interaction.response.send_message(embed=embed, view=view)


# Register the persistent views for all known temp channels on startup
# This is handled by setup_hook overriding
_original_setup_hook = CodeBot.setup_hook


async def _patched_setup_hook(self):
    await _original_setup_hook(self)
    # Register temp voice command groups
    self.tree.add_command(tempvoice_group)
    self.tree.add_command(voice_group)
    # Register persistent views for existing temp channels
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT channel_id, privacy FROM temp_voice_active")
    for row in cur.fetchall():
        self.add_view(TempVoiceInterface(int(row[0]), row[1] or "public"))
    conn.close()


CodeBot.setup_hook = _patched_setup_hook


_mc_bot = MCBotManager()

_mc_mc_group = discord.app_commands.Group(name="mc", description="Minecraft bot commands")

@_mc_mc_group.command(name="join", description="Join a Minecraft server")
@app_commands.describe(host="Server IP or domain", port="Port (default 25565)", username="Bot username")
async def mc_join(interaction: discord.Interaction, host: str, port: int = 25565, username: str = "NullBot"):
    await interaction.response.defer()
    async def _chat_relay(sender: str, message: str):
        if _mc_bot._mc_channel_id:
            ch = bot.get_channel(_mc_bot._mc_channel_id)
            if ch:
                await ch.send(f"**[{sender}]** {message}")
    result = await _mc_bot.connect(host, port, username, _chat_relay, interaction.channel_id)
    await interaction.followup.send(result)

@_mc_mc_group.command(name="leave", description="Leave the Minecraft server")
async def mc_leave(interaction: discord.Interaction):
    await _mc_bot.disconnect()
    await interaction.response.send_message(":door: Left the server.")

@_mc_mc_group.command(name="say", description="Send a chat message in Minecraft")
@app_commands.describe(message="Message to send")
async def mc_say(interaction: discord.Interaction, message: str):
    if not _mc_bot.connected:
        await interaction.response.send_message(":x: Not connected to a Minecraft server")
        return
    await _mc_bot.send_chat(f"[Discord] {interaction.user.display_name}: {message}")
    await interaction.response.send_message(f":speech_balloon: Sent: {message}")

@_mc_mc_group.command(name="follow", description="Follow a player")
@app_commands.describe(target="Player name to follow")
async def mc_follow(interaction: discord.Interaction, target: str):
    await interaction.response.defer()
    result = await _mc_bot.follow(target)
    await interaction.followup.send(result)

@_mc_mc_group.command(name="come", description="Make the bot come to your position (use coordinates)")
@app_commands.describe(x="X coordinate", y="Y coordinate", z="Z coordinate")
async def mc_come(interaction: discord.Interaction, x: float, y: float, z: float):
    await interaction.response.defer()
    result = await _mc_bot.come_to(x, y, z)
    await interaction.followup.send(result)

@_mc_mc_group.command(name="goto", description="Move to coordinates")
@app_commands.describe(x="X coordinate", y="Y coordinate", z="Z coordinate")
async def mc_goto(interaction: discord.Interaction, x: float, y: float, z: float):
    await interaction.response.defer()
    result = await _mc_bot.goto(x, y, z)
    await interaction.followup.send(result)

@_mc_mc_group.command(name="stop", description="Stop all actions (follow, etc.)")
async def mc_stop(interaction: discord.Interaction):
    result = await _mc_bot.stop()
    await interaction.response.send_message(result)

@_mc_mc_group.command(name="attack", description="Attack a nearby player")
@app_commands.describe(target="Player name to attack")
async def mc_attack(interaction: discord.Interaction, target: str):
    await interaction.response.defer()
    result = await _mc_bot.attack(target)
    await interaction.followup.send(result)

@_mc_mc_group.command(name="mine", description="Mine the block you're looking at")
async def mc_mine(interaction: discord.Interaction):
    await interaction.response.defer()
    result = await _mc_bot.mine()
    await interaction.followup.send(result)

@_mc_mc_group.command(name="pos", description="Show the bot's current position")
async def mc_pos(interaction: discord.Interaction):
    pos = _mc_bot.get_position()
    if pos:
        x, y, z = pos
        await interaction.response.send_message(f":military_helmet: Position: **{x:.1f}, {y:.1f}, {z:.1f}**")
    else:
        await interaction.response.send_message(":x: Not connected")

bot.tree.add_command(_mc_mc_group)

# =============================================================================
# Welcome / Goodbye / Autorole System (like Welcomer.gg)
# =============================================================================

import io as _io
from PIL import Image, ImageDraw, ImageFont, ImageFilter as _ImageFilter
import textwrap as _textwrap

def _get_welcome_settings(guild_id: int) -> dict | None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT channel_id, message, image_url, enabled FROM welcome_settings WHERE guild_id = ?", (guild_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"channel_id": row[0], "message": row[1], "image_url": row[2], "enabled": row[3]}

def _get_goodbye_settings(guild_id: int) -> dict | None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT channel_id, message, enabled FROM goodbye_settings WHERE guild_id = ?", (guild_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    return {"channel_id": row[0], "message": row[1], "enabled": row[2]}

def _get_autoroles(guild_id: int) -> list[int]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT role_id FROM autoroles WHERE guild_id = ?", (guild_id,))
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows] if rows else []

def _get_welcome_dm(guild_id: int) -> dict:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT enabled, message FROM welcome_dm_settings WHERE guild_id = ?", (guild_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"enabled": 0, "message": "Welcome to **{server}**! Check out the rules in #rules!"}
    return {"enabled": row[0], "message": row[1]}

def _format_welcome_text(template: str, member: discord.Member, guild: discord.Guild) -> str:
    count = guild.member_count or 0
    text = template.replace("{user}", member.mention)
    text = text.replace("{user.name}", member.name)
    text = text.replace("{user.tag}", str(member))
    text = text.replace("{server}", guild.name)
    text = text.replace("{count}", str(count))
    suff = "th" if 11 <= count % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(count % 10, "th")
    text = text.replace("{count.ordinal}", f"{count}{suff}")
    return text

async def _generate_welcome_image(member: discord.Member, guild: discord.Guild, bg_url: str | None = None) -> _io.BytesIO | None:
    try:
        import aiohttp
        width, height = 800, 400
        img = Image.new("RGBA", (width, height), (30, 30, 40, 255))
        draw = ImageDraw.Draw(img)

        if bg_url:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(bg_url, timeout=10) as resp:
                        if resp.status == 200:
                            bg_data = await resp.read()
                            bg = Image.open(_io.BytesIO(bg_data)).convert("RGBA")
                            bg = bg.resize((width, height), Image.LANCZOS)
                            img = Image.blend(bg, img, 0.4)
                            draw = ImageDraw.Draw(img)
            except Exception as e:
                print(f"[WELCOME IMG] bg fetch failed: {e}")

        # Gradient overlay
        for y in range(height):
            alpha = int(80 * (1 - y / height))
            draw.rectangle([(0, y), (width, y)], fill=(0, 0, 0, alpha))

        font_large = None
        font_small = None
        font_med = None
        for size, attr in ((60, "font_large"), (28, "font_med"), (22, "font_small")):
            try:
                f = ImageFont.truetype("arial.ttf", size)
            except Exception:
                try:
                    f = ImageFont.truetype("DejaVuSans.ttf", size)
                except Exception:
                    f = ImageFont.load_default()
            if attr == "font_large":
                font_large = f
            elif attr == "font_med":
                font_med = f
            else:
                font_small = f

        # Avatar
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(str(member.display_avatar.url), timeout=10) as resp:
                    if resp.status == 200:
                        av_data = await resp.read()
                        av = Image.open(_io.BytesIO(av_data)).convert("RGBA")
                        av = av.resize((160, 160), Image.LANCZOS)
                        mask = Image.new("L", (160, 160), 0)
                        mask_draw = ImageDraw.Draw(mask)
                        mask_draw.ellipse((0, 0, 160, 160), fill=255)
                        img.paste(av, (width // 2 - 80, 40), mask)
        except Exception as e:
            print(f"[WELCOME IMG] avatar failed: {e}")

        # Welcome text
        welcome_text = "WELCOME"
        if font_large:
            bbox = draw.textbbox((0, 0), welcome_text, font=font_large)
            tw = bbox[2] - bbox[0]
            draw.text(((width - tw) / 2, 215), welcome_text, fill=(255, 255, 255, 255), font=font_large)

        # Username
        name_text = member.name
        if font_med:
            bbox = draw.textbbox((0, 0), name_text, font=font_med)
            tw = bbox[2] - bbox[0]
            draw.text(((width - tw) / 2, 275), name_text, fill=(180, 220, 255, 255), font=font_med)

        # Server name and member count
        sub_text = f"{guild.name}  |  Member #{guild.member_count}"
        if font_small:
            bbox = draw.textbbox((0, 0), sub_text, font=font_small)
            tw = bbox[2] - bbox[0]
            draw.text(((width - tw) / 2, 320), sub_text, fill=(200, 200, 210, 255), font=font_small)

        buf = _io.BytesIO()
        img.save(buf, "PNG")
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"[WELCOME IMG] error: {e}")
        return None


@bot.event
async def on_member_join(member: discord.Member):
    guild = member.guild
    settings = _get_welcome_settings(guild.id)
    if not settings or not settings["enabled"]:
        return
    channel = guild.get_channel(settings["channel_id"]) if settings["channel_id"] else None
    if not channel:
        return

    text = _format_welcome_text(settings["message"], member, guild)
    image_url = settings.get("image_url") or None
    img_buf = await _generate_welcome_image(member, guild, image_url)

    try:
        if img_buf:
            await channel.send(text, file=discord.File(img_buf, "welcome.png"))
        else:
            await channel.send(text)
    except Exception as e:
        print(f"[WELCOME] send error: {e}")

    # Autorole
    for role_id in _get_autoroles(guild.id):
        role = guild.get_role(role_id)
        if role:
            try:
                await member.add_roles(role, reason="Autorole")
            except Exception as e:
                print(f"[AUTOROLE] failed for {member.name}: {e}")

    # Welcome DM
    dm = _get_welcome_dm(guild.id)
    if dm["enabled"]:
        try:
            dm_text = _format_welcome_text(dm["message"], member, guild)
            await member.send(dm_text)
        except Exception as e:
            print(f"[WELCOME DM] failed for {member.name}: {e}")


@bot.event
async def on_member_remove(member: discord.Member):
    guild = member.guild
    settings = _get_goodbye_settings(guild.id)
    if not settings or not settings["enabled"]:
        return
    channel = guild.get_channel(settings["channel_id"]) if settings["channel_id"] else None
    if not channel:
        return

    text = _format_welcome_text(settings["message"], member, guild)
    try:
        await channel.send(text)
    except Exception as e:
        print(f"[GOODBYE] send error: {e}")


_welcome_group = app_commands.Group(name="welcome", description="Set up welcome messages with images", guild_only=True)


@_welcome_group.command(name="channel", description="Set the welcome message channel")
@app_commands.describe(channel="The channel to post welcome messages in")
async def welcome_channel(interaction: discord.Interaction, channel: discord.TextChannel):
    conn = get_db()
    conn.execute("INSERT INTO welcome_settings (guild_id, channel_id, message, enabled) VALUES (?, ?, 'Welcome {user} to **{server}**! You are member **#{count}**!', 1) ON CONFLICT(guild_id) DO UPDATE SET channel_id = ?", (interaction.guild_id, channel.id, channel.id))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: Welcome messages will be sent to {channel.mention}")


@_welcome_group.command(name="message", description="Set the welcome message text")
@app_commands.describe(text='Message text. Use {user}, {user.name}, {server}, {count}, {count.ordinal}')
async def welcome_message(interaction: discord.Interaction, text: str):
    conn = get_db()
    conn.execute("INSERT INTO welcome_settings (guild_id, channel_id, message, enabled) VALUES (?, NULL, ?, 1) ON CONFLICT(guild_id) DO UPDATE SET message = ?", (interaction.guild_id, text, text))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: Welcome message set!\n```{text}```")


@_welcome_group.command(name="image", description="Set a background image URL for the welcome card")
@app_commands.describe(url="Direct image URL for the background. Leave empty to reset to default.")
async def welcome_image(interaction: discord.Interaction, url: str = None):
    conn = get_db()
    if url:
        conn.execute("INSERT INTO welcome_settings (guild_id, channel_id, message, image_url, enabled) VALUES (?, NULL, 'Welcome {user} to **{server}**!', ?, 1) ON CONFLICT(guild_id) DO UPDATE SET image_url = ?", (interaction.guild_id, url, url))
        await interaction.response.send_message(f":frame_photo: Welcome card background set!")
    else:
        conn.execute("INSERT INTO welcome_settings (guild_id, channel_id, message, enabled) VALUES (?, NULL, 'Welcome {user} to **{server}**!', 1) ON CONFLICT(guild_id) DO UPDATE SET image_url = NULL", (interaction.guild_id,))
        await interaction.response.send_message(":frame_photo: Welcome card background reset to default.")
    conn.commit()
    conn.close()


@_welcome_group.command(name="toggle", description="Enable or disable welcome messages")
@app_commands.describe(enabled="True to enable, False to disable")
async def welcome_toggle(interaction: discord.Interaction, enabled: bool):
    conn = get_db()
    conn.execute("INSERT INTO welcome_settings (guild_id, channel_id, message, enabled) VALUES (?, NULL, 'Welcome {user} to **{server}**!', ?) ON CONFLICT(guild_id) DO UPDATE SET enabled = ?", (interaction.guild_id, int(enabled), int(enabled)))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: Welcome messages **{'enabled' if enabled else 'disabled'}**")


@_welcome_group.command(name="test", description="Preview the welcome message and card")
async def welcome_test(interaction: discord.Interaction):
    await interaction.response.defer()
    settings = _get_welcome_settings(interaction.guild_id)
    text = "Welcome to the server!"
    if settings:
        text = _format_welcome_text(settings["message"], interaction.user, interaction.guild)
    img_buf = await _generate_welcome_image(interaction.user, interaction.guild, settings.get("image_url") if settings else None)
    if img_buf:
        await interaction.followup.send(text, file=discord.File(img_buf, "welcome.png"))
    else:
        await interaction.followup.send(text)


bot.tree.add_command(_welcome_group)


_goodbye_group = app_commands.Group(name="goodbye", description="Set up goodbye messages", guild_only=True)


@_goodbye_group.command(name="channel", description="Set the goodbye message channel")
@app_commands.describe(channel="The channel to post goodbye messages in")
async def goodbye_channel(interaction: discord.Interaction, channel: discord.TextChannel):
    conn = get_db()
    conn.execute("INSERT INTO goodbye_settings (guild_id, channel_id, message, enabled) VALUES (?, ?, 'Goodbye {user.name}, we will miss you!', 1) ON CONFLICT(guild_id) DO UPDATE SET channel_id = ?", (interaction.guild_id, channel.id, channel.id))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: Goodbye messages will be sent to {channel.mention}")


@_goodbye_group.command(name="message", description="Set the goodbye message text")
@app_commands.describe(text='Message text. Use {user}, {user.name}, {server}')
async def goodbye_message(interaction: discord.Interaction, text: str):
    conn = get_db()
    conn.execute("INSERT INTO goodbye_settings (guild_id, channel_id, message, enabled) VALUES (?, NULL, ?, 1) ON CONFLICT(guild_id) DO UPDATE SET message = ?", (interaction.guild_id, text, text))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: Goodbye message set!\n```{text}```")


@_goodbye_group.command(name="toggle", description="Enable or disable goodbye messages")
@app_commands.describe(enabled="True to enable, False to disable")
async def goodbye_toggle(interaction: discord.Interaction, enabled: bool):
    conn = get_db()
    conn.execute("INSERT INTO goodbye_settings (guild_id, channel_id, message, enabled) VALUES (?, NULL, 'Goodbye {user.name}!', ?) ON CONFLICT(guild_id) DO UPDATE SET enabled = ?", (interaction.guild_id, int(enabled), int(enabled)))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: Goodbye messages **{'enabled' if enabled else 'disabled'}**")


bot.tree.add_command(_goodbye_group)


_welcomedm_group = app_commands.Group(name="welcomedm", description="Set up welcome DMs for new members", guild_only=True)


@_welcomedm_group.command(name="toggle", description="Enable or disable welcome DMs")
@app_commands.describe(enabled="True to enable, False to disable")
async def welcomedm_toggle(interaction: discord.Interaction, enabled: bool):
    conn = get_db()
    conn.execute("INSERT INTO welcome_dm_settings (guild_id, enabled, message) VALUES (?, ?, 'Welcome to **{server}**!') ON CONFLICT(guild_id) DO UPDATE SET enabled = ?", (interaction.guild_id, int(enabled), int(enabled)))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: Welcome DMs **{'enabled' if enabled else 'disabled'}**")


@_welcomedm_group.command(name="message", description="Set the welcome DM text")
@app_commands.describe(text='Message text. Use {user}, {user.name}, {server}')
async def welcomedm_message(interaction: discord.Interaction, text: str):
    conn = get_db()
    conn.execute("INSERT INTO welcome_dm_settings (guild_id, enabled, message) VALUES (?, 0, ?) ON CONFLICT(guild_id) DO UPDATE SET message = ?", (interaction.guild_id, text, text))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: Welcome DM message set!\n```{text}```")


bot.tree.add_command(_welcomedm_group)


_autorole_group = app_commands.Group(name="autorole", description="Manage auto-assign roles for new members", guild_only=True)


@_autorole_group.command(name="add", description="Add a role that gets auto-assigned to new members")
@app_commands.describe(role="The role to auto-assign")
async def autorole_add(interaction: discord.Interaction, role: discord.Role):
    conn = get_db()
    conn.execute("INSERT OR IGNORE INTO autoroles (guild_id, role_id) VALUES (?, ?)", (interaction.guild_id, role.id))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":white_check_mark: **{role.name}** will be auto-assigned to new members.")


@_autorole_group.command(name="remove", description="Stop auto-assigning a role to new members")
@app_commands.describe(role="The role to remove from auto-assign")
async def autorole_remove(interaction: discord.Interaction, role: discord.Role):
    conn = get_db()
    conn.execute("DELETE FROM autoroles WHERE guild_id = ? AND role_id = ?", (interaction.guild_id, role.id))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f":wastebasket: **{role.name}** will no longer be auto-assigned.")


@_autorole_group.command(name="list", description="List all auto-assign roles")
async def autorole_list(interaction: discord.Interaction):
    role_ids = _get_autoroles(interaction.guild_id)
    if not role_ids:
        await interaction.response.send_message(":information_source: No auto-assign roles configured.")
        return
    roles = [interaction.guild.get_role(rid) for rid in role_ids]
    role_mentions = [r.mention for r in roles if r]
    await interaction.response.send_message(f":scroll: Auto-assign roles:\n" + "\n".join(f"- {r}" for r in role_mentions))


bot.tree.add_command(_autorole_group)

# =============================================================================
# Counting Game
# =============================================================================

_counting_group = app_commands.Group(name="counting", description="Counting game — take turns counting up!", guild_only=True)


@_counting_group.command(name="setup", description="Set the counting channel")
@app_commands.describe(channel="Channel for counting")
async def counting_setup(interaction: discord.Interaction, channel: discord.TextChannel):
    conn = get_db()
    conn.execute("INSERT INTO counting_settings (guild_id, channel_id) VALUES (?, ?) ON CONFLICT(guild_id) DO UPDATE SET channel_id = ?",
                 (interaction.guild_id, channel.id, channel.id))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f"\U0001f522 Counting channel set to {channel.mention}!")


@_counting_group.command(name="stats", description="Show counting stats for this server")
async def counting_stats(interaction: discord.Interaction):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT current_count, highest_count FROM counting_stats WHERE guild_id = ?", (interaction.guild_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        await interaction.response.send_message(":information_source: No counting has happened yet! Use `/counting setup` first.")
        return
    embed = discord.Embed(title="\U0001f522 Counting Stats", color=0x2ecc71)
    embed.add_field(name="Current Count", value=str(row[0]), inline=True)
    embed.add_field(name="Highest Ever", value=str(row[1]), inline=True)
    await interaction.response.send_message(embed=embed)


@_counting_group.command(name="reset", description="Reset the count to 0 (Admin only)")
async def counting_reset(interaction: discord.Interaction):
    conn = get_db()
    conn.execute("INSERT INTO counting_stats (guild_id, current_count, highest_count, last_user_id) VALUES (?, 0, 0, 0) ON CONFLICT(guild_id) DO UPDATE SET current_count = 0, last_user_id = 0",
                 (interaction.guild_id,))
    conn.commit()
    conn.close()
    await interaction.response.send_message(":arrows_counterclockwise: Count reset to **0**. Highest score preserved.")


bot.tree.add_command(_counting_group)


bot.run(TOKEN)
