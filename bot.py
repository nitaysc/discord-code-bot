import asyncio
import io
import os
import random
import re
import sqlite3
import tempfile
import textwrap
from collections import deque
from datetime import datetime, timezone

import aiohttp
import discord
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from openai import OpenAI
import wavelink
from ddgs import DDGS
from PIL import Image, ImageDraw, ImageFont

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
CLOUDFLARE_KEY = os.getenv("CLOUDFLARE_API_KEY")
CLOUDFLARE_ACCOUNT = os.getenv("CLOUDFLARE_ACCOUNT_ID")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
AI_KEY = GITHUB_TOKEN or OPENROUTER_KEY or HF_TOKEN or CLOUDFLARE_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("AI_MODEL", "gpt-4o-mini")
if MODEL.startswith("AI_MODEL="):
    MODEL = MODEL[len("AI_MODEL="):]

if not TOKEN or not AI_KEY:
    raise RuntimeError("Missing DISCORD_TOKEN or AI API key in .env file")

print(f"[STARTUP] AI provider: {MODEL}")
print(f"[STARTUP] Search keys detected: SerpApi={'yes' if os.getenv('SERPAPI_API_KEY') else 'no'}, Bing={'yes' if os.getenv('BING_API_KEY') else 'no'}, Brave={'yes' if os.getenv('BRAVE_API_KEY') else 'no'}")

if GITHUB_TOKEN:
    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=GITHUB_TOKEN,
    )
elif OPENROUTER_KEY:
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_KEY,
    )
elif HF_TOKEN:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )
elif CLOUDFLARE_KEY and CLOUDFLARE_ACCOUNT:
    client = OpenAI(
        base_url=f"https://api.cloudflare.com/client/v4/accounts/{CLOUDFLARE_ACCOUNT}/ai/v1/",
        api_key=CLOUDFLARE_KEY,
    )
elif os.getenv("OPENAI_API_KEY"):
    client = OpenAI(api_key=AI_KEY)
elif os.getenv("GROQ_API_KEY"):
    client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=AI_KEY,
    )
else:
    client = OpenAI(
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key=AI_KEY,
    )


def _search_duckduckgo(query: str, max_results: int = 5) -> list[dict]:
    try:
        results = list(DDGS().text(query, max_results=max_results))
        if not results:
            results = list(DDGS().news(query, max_results=max_results))
        return results
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
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


SEARCH_TRIGGER_WORDS = {
    "latest", "current", "today", "now", "news", "weather", "price", "prices",
    "score", "scores", "update", "recent", "happened", "happening", "live",
    "stock", "crypto", "bitcoin", "election", "who won", "who won", "winner", "winners", "results",
    "release date", "when did", "how old is", "age of", "net worth", "did happen", "has happened",
    "look online", "look up", "search for", "find online", "check online",
    "what is", "how to", "where is", "who is", "who was",
    "world cup", "olympics", "super bowl", "champions league", "eurovision",
    "2025", "2026", "2027", "yesterday", "tomorrow", "this week", "last week",
}


def should_search(question: str) -> tuple[bool, str]:
    lowered = question.lower()
    if any(word in lowered for word in SEARCH_TRIGGER_WORDS):
        return True, question
    return False, ""


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

DB_PATH = os.path.join(os.path.dirname(__file__), "bot.db")


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS levels (
            guild_id INTEGER NOT NULL,
            user_id INTEGER NOT NULL,
            xp INTEGER DEFAULT 0,
            messages INTEGER DEFAULT 0,
            voice_minutes INTEGER DEFAULT 0,
            last_xp REAL DEFAULT 0,
            PRIMARY KEY (guild_id, user_id)
        );
        CREATE TABLE IF NOT EXISTS level_settings (
            guild_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT,
            PRIMARY KEY (guild_id, key)
        );
        CREATE TABLE IF NOT EXISTS role_rewards (
            guild_id INTEGER NOT NULL,
            level INTEGER NOT NULL,
            role_id INTEGER NOT NULL,
            PRIMARY KEY (guild_id, level)
        );
        CREATE TABLE IF NOT EXISTS xp_blacklist (
            guild_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            target_type TEXT NOT NULL,
            PRIMARY KEY (guild_id, target_id, target_type)
        );
        CREATE TABLE IF NOT EXISTS xp_multipliers (
            guild_id INTEGER NOT NULL,
            target_id INTEGER NOT NULL,
            target_type TEXT NOT NULL,
            multiplier REAL NOT NULL,
            PRIMARY KEY (guild_id, target_id, target_type)
        );
        """
    )
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
        return None
    now = datetime.now(timezone.utc).timestamp()
    key = (guild_id, user_id)
    cooldown = float(_get_setting(guild_id, "cooldown", 30))
    if key in _xp_cooldowns and now - _xp_cooldowns[key] < cooldown:
        return None
    _xp_cooldowns[key] = now
    min_xp = int(_get_setting(guild_id, "min_xp", 15))
    max_xp = int(_get_setting(guild_id, "max_xp", 25))
    gained = random.randint(min_xp, max_xp)
    mult = _get_multiplier(guild_id, channel_id, role_ids)
    gained = int(gained * mult)
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
                 xp = xp + ?,
                 messages = messages + 1,
                 last_xp = ?""",
            (guild_id, user_id, new_xp, now, amount, now),
        )
    else:
        cur.execute(
            """INSERT INTO levels (guild_id, user_id, xp, voice_minutes, last_xp)
               VALUES (?, ?, ?, 1, ?)
               ON CONFLICT(guild_id, user_id) DO UPDATE SET
                 xp = xp + ?,
                 voice_minutes = voice_minutes + 1,
                 last_xp = ?""",
            (guild_id, user_id, new_xp, now, amount, now),
        )
    conn.commit()
    conn.close()
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
    enabled = _get_setting(member.guild.id, "levelup_enabled", "true").lower()
    if enabled != "true":
        return
    levelup_channel_id = _get_setting(member.guild.id, "levelup_channel")
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
MY CAPABILITIES:
- Play music: /play <song name>, /play <YouTube URL>. Also /skip /stop /queue /pause /resume /volume.
- Search the web: /search <query>, or ask me anything and I will search if I need current info.
- Leveling system: /rank, /leaderboard, XP for messages and voice, role rewards, multipliers, blacklists.
- Generate ANY file: /lua, /script, /file, or just say "make me a .file type..."
- Create .exe source with compile instructions
- Read & summarize channels: /read #channel
- See who's in voice: /voice or ask "who's in vc?"
- Kick/vkick/say/clear: admin commands
- Normal chat, coding help, answering questions
- See images: attach an image and ask about it
- Read files: drop a .lua, .txt, .py, .json, or any text file and ask about it
- Voice features coming soon (speech-to-text and text-to-speech)
- I remember the last 50 messages in each channel

I am a real Discord bot with real features. Never say "I can't" without checking my actual capabilities above.
"""

CHAT_SYSTEM = textwrap.dedent(f"""\
You are Null, a friendly and capable Discord bot. You are chill, concise, and cool.
Today's date: {datetime.now(timezone.utc).strftime('%Y-%m-%d')}.
{CHAT_CAPABILITIES}

Rules:
- When someone asks if you can do something, check your capabilities FIRST before saying no.
- Guide users to use the right /command for what they want.
- For code/file requests, output the content and the system sends it as a downloadable file.
- For ANY question about current events, recent news, sports results, today's date, future dates, or anything time-sensitive: you MUST rely on the web search results provided in the prompt, NOT your training data. Your training data has a cutoff and may be outdated.
- If web search results are provided, use them as the authoritative source. Do not contradict them with your built-in knowledge.
- If no search results are provided and the question is time-sensitive, say you don't have current info rather than guessing.
- Keep responses short and natural. Don't list all features unless asked.
- Respond in the same language the user speaks.
""")

CODE_SYSTEM = textwrap.dedent("""\
You are a coding assistant. Write code based on the user's request.
If the request is vague or missing details, use sensible defaults and write the code anyway.
Do NOT ask clarifying questions. Do NOT explain what details are needed.
Output ONLY the code in a code block with language tag. No explanations outside the code block.
Follow best practices and proper syntax.
""")

YIMMENU_LUA_SYSTEM = textwrap.dedent("""\
You are a YimMenuV2 Lua scripting assistant for GTA V Enhanced.
Write Lua scripts that work with the YimMenuV2 mod menu.
Do NOT ask clarifying questions. Use sensible defaults and write complete, working code.
Output ONLY the code in a code block with `lua` language tag. No explanations outside the code block.

CRITICAL API RULES — never violate these:
- natives.load_natives() takes NO arguments. Never write natives.load_natives(number).
- commandmgr.add_looped_command takes 6 arguments: id, label, description, tick_function, on_enable_function, on_disable_function. Always include the on_enable and on_disable callbacks.
- Handles returned by GET_VEHICLE_PED_IS_IN or PLAYER_PED_ID can be 0. Check with `if handle ~= 0 then`, never `if handle then`.
- ENTITY.SET_ENTITY_VELOCITY takes separate x, y, z numbers: SET_ENTITY_VELOCITY(ent, x, y, z). Never pass a Vector3 object.
- For a true instant vehicle stop, always use BOTH `VEHICLE.SET_VEHICLE_FORWARD_SPEED(veh, 0)` AND `ENTITY.SET_ENTITY_VELOCITY(veh, 0, 0, 0)` together.
- PED.GET_VEHICLE_PED_IS_IN(ped, lastVehicle) returns the vehicle handle.
- PED.IS_PED_IN_ANY_VEHICLE(ped, atGetIn) returns true/false.
- For interactive toggle/button scripts, always create a menu: menu.set_menu_name, menu.get_submenu, add_category, add_group, then add the command to the group with group:add_command(id).

Known YimMenuV2 Lua API:
- natives.load_natives() — call once at the top if using GTA natives.
- menu.set_menu_name("My Menu") / menu.get_submenu("My Menu") / submenu:add_category / category:add_group / group:add_command / group:add_button
- commandmgr.add_command(id, label, desc, callback)
- commandmgr.add_looped_command(id, label, desc, tick_fn, on_enable_fn, on_disable_fn)
- commandmgr.add_list_command(id, label, desc, {{1,"Option"}, ...}, default_index, callback)
- script.run_in_callback(function() ... end) / script.yield(ms)
- log.info / notify.success / notify.info / notify.error / notify.warning
- stats.get_int / stats.set_int / stats.set_bool
- util.joaat("MODEL_NAME")
- PLAYER.PLAYER_PED_ID / PLAYER.PLAYER_ID
- ENTITY.GET_ENTITY_COORDS(ent, alive) / GET_ENTITY_MODEL / SET_ENTITY_HEALTH / SET_ENTITY_VELOCITY(ent, x, y, z)
- VEHICLE.CREATE_VEHICLE / SET_VEHICLE_ENGINE_ON / SET_VEHICLE_FORWARD_SPEED
- PED.CREATE_PED / SET_PED_TO_RAGDOLL / IS_PED_IN_ANY_VEHICLE / GET_VEHICLE_PED_IS_IN
- STREAMING.REQUEST_MODEL / HAS_MODEL_LOADED / SET_MODEL_AS_NO_LONGER_NEEDED
- entities.get_all_peds_as_handles / Entity(handle) / Vector3(x,y,z) / FIRE.ADD_EXPLOSION

Filenaming:
- Put `-- filename: short_snake_case_name.lua` as the very first line of the Lua code.
- Example: `-- filename: instant_car_brake.lua`

Example structure for a complete interactive toggle script:
```lua
-- filename: example_script.lua
natives.load_natives()

menu.set_menu_name("My Script")
local submenu = menu.get_submenu("My Script")
local category = submenu:add_category("Actions")
local group = category:add_group("Toggles")

local function tick_fn()
    -- loop logic
    script.yield(0)
end

commandmgr.add_looped_command("my_toggle", "My Toggle", "Does something", tick_fn,
    function() notify.success("My Toggle", "Enabled") end,
    function() notify.success("My Toggle", "Disabled") end)

group:add_command("my_toggle")
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
        "can you", "could you", "would you", "will you",
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
    ]
    return any(ind in text_lower for ind in file_indicators)


def choose_filename(file_type: str | None, content: str, index: int = 0) -> str:
    ext = FILE_EXTENSIONS.get(file_type, ".txt") if file_type else ".txt"
    lines = content.strip().split("\n") if content else []
    # Look for an explicit filename comment in the first 5 lines
    for line in lines[:5]:
        m = re.match(r"^\s*--\s*filename\s*:\s*(\S+)(\.\w+)?\s*$", line, re.IGNORECASE)
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


def _call_ai(system: str, prompt: str, history: list[dict] | None = None,
              temperature: float = 0.5, max_tokens: int = 4096,
              image_urls: list[str] | None = None) -> str:
    messages = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)

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

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def call_ai(system: str, prompt: str, history: list[dict] | None = None,
                  temperature: float = 0.5, max_tokens: int = 4096,
                  image_urls: list[str] | None = None) -> str:
    return await asyncio.to_thread(_call_ai, system, prompt, history, temperature, max_tokens, image_urls)


async def answer_with_web_search_if_needed(
    prompt: str,
    history: list[dict] | None = None,
    image_urls: list[str] | None = None,
    temperature: float = 0.7,
) -> str:
    needs_search, search_query = should_search(prompt)
    if needs_search:
        search_results = ""
        for query in [search_query, " ".join(
            w for w in search_query.split()
            if w.lower() not in {"how", "to", "a", "the", "is", "are", "for", "of", "in", "on", "and", "or", "i", "you", "me", "tell", "something", "look", "online", "search", "find", "check"}
        )]:
            if not query or query == search_results:
                continue
            search_results = await web_search(query, max_results=5)
            if search_results and not search_results.startswith("Search error:"):
                break
            search_results = ""
        if search_results:
            enhanced_prompt = (
                f"{prompt}\n\n[Web search results for '{search_query}':\n"
                f"{search_results}\n\n"
                f"Use the above search results as the authoritative source."
            )
            return await call_ai(CHAT_SYSTEM, enhanced_prompt, history, temperature, image_urls=image_urls)
        return (
            "I searched online but couldn't find current results right now. "
            "Free web search may be blocked on this host. "
            "For reliable real-time info, add a search API key: SERPAPI_API_KEY, BING_API_KEY, or BRAVE_API_KEY."
        )
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
            wavelink.Node(uri="http://lavalink.jirayu.net:13592", password="youshallnotpass"),
            wavelink.Node(uri="http://lavalinkv4.serenetia.com:80", password="https://seretia.link/discord"),
            wavelink.Node(uri="http://lava.g3v.co.uk:9008", password="lavalinklol"),
            wavelink.Node(uri="http://lavalink.triniumhost.com:4333", password="free"),
            wavelink.Node(uri="http://n3.nexcloud.in:2026", password="nexcloud"),
        ]
        for node in nodes:
            try:
                await wavelink.Pool.connect(nodes=[node], client=self)
                print(f"Wavelink connected: {node.uri}")
                break
            except Exception as e:
                print(f"Wavelink failed {node.uri}: {e}")
        else:
            print("WARNING: Could not connect to any public Lavalink node.")
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
        for guild in self.guilds:
            try:
                self.tree.clear_commands(guild=guild)
                self.tree.copy_global_to(guild=guild)
                await self.tree.sync(guild=guild)
                print(f"Synced commands to guild: {guild.name}")
            except Exception as e:
                print(f"Failed to sync to {guild.name}: {e}")
        try:
            self.tree.clear_commands(guild=None)
            await self.tree.sync()
            print("Cleared global commands to avoid duplicates")
        except Exception as e:
            print(f"Failed to clear global commands: {e}")

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
                        if _is_blacklisted(guild.id, vc.id, role_ids):
                            continue
                        mult = _get_multiplier(guild.id, vc.id, role_ids)
                        amount = int(10 * mult)
                        result = add_voice_xp(guild.id, member.id, amount)
                        if result:
                            xp, level, leveled_up = result
                            if leveled_up:
                                text_ch = guild.system_channel
                                if text_ch:
                                    await handle_level_up(member, text_ch, level)
        except Exception as e:
            print(f"Voice XP task error: {e}")


bot = CodeBot()


@bot.event
async def on_message(message):
    if message.author.bot:
        return

    is_mention = bot.user is not None and (
        bot.user in message.mentions
        or f"<@{bot.user.id}>" in message.content
        or f"<@!{bot.user.id}>" in message.content
    )
    is_dm = isinstance(message.channel, discord.DMChannel)

    if is_mention or is_dm:
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
            if att.content_type and att.content_type.startswith("image/"):
                image_urls.append(att.url)
            else:
                file_data = await read_attachment(att)
                if file_data:
                    filename, file_text = file_data
                    file_contexts.append(f"[Attached file `{filename}`:\n```\n{file_text}\n```]")

        channel_id = message.channel.id
        display_name = message.author.display_name
        history_entry = f"{display_name}: {content}"
        if image_urls:
            history_entry += f" [attached {len(image_urls)} image(s)]"
        if file_contexts:
            history_entry += f" [attached {len(file_contexts)} file(s)]"
        add_to_history(channel_id, "user", history_entry)

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
                    if message.guild and any(w in content_lower for w in ["voice", "vc", "vchat", "voice chat", "talk"]):
                        context_extra = f"\n\n[Server voice info: {get_voice_info(message.guild)}]"
                    if file_extra:
                        context_extra += f"\n\n{file_extra}"

                    history = get_history(channel_id)
                    prompt = content + context_extra
                    answer = await answer_with_web_search_if_needed(
                        prompt, history, image_urls=image_urls if image_urls else None
                    )
                    add_to_history(channel_id, "assistant", answer)
                    code_blocks = extract_code_blocks(answer)
                    await message.reply(answer, mention_author=False)
                    if code_blocks:
                        await send_files(message.channel, code_blocks)
            except Exception as e:
                await message.reply(f":x: Error: {e}", mention_author=False)
        return

    if message.guild:
        role_ids = [r.id for r in message.author.roles]
        result = add_message_xp(message.guild.id, message.author.id, message.channel.id, role_ids)
        if result:
            xp, level, leveled_up = result
            if leveled_up:
                await handle_level_up(message.author, message.channel, level)

    await bot.process_commands(message)


@bot.tree.command(name="rank", description="Check your XP and level")
@app_commands.describe(member="Whose rank to check (default: you)")
async def slash_rank(interaction: discord.Interaction, member: discord.Member = None):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    target = member or interaction.user
    data = get_rank(interaction.guild.id, target.id)
    if not data:
        await interaction.response.send_message(
            f":x: {target.mention} has no XP yet. Start chatting!", ephemeral=True
        )
        return
    embed = discord.Embed(
        title=f":chart_with_upwards_trend: {target.display_name}'s Rank",
        color=discord.Color.blue(),
    )
    embed.set_thumbnail(url=target.display_avatar.url)
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
        await interaction.response.send_message(embed=embed, file=discord.File(card, filename="rank.png"))
    except Exception as e:
        print(f"Rank card error: {e}")
        await interaction.response.send_message(embed=embed)


@bot.tree.command(name="leaderboard", description="Top 10 most active members")
async def slash_leaderboard(interaction: discord.Interaction):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    rows = get_leaderboard(interaction.guild.id, 10)
    if not rows:
        await interaction.response.send_message(":x: No one has XP yet.")
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
    await interaction.response.send_message(embed=embed)


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
            answer = await answer_with_web_search_if_needed(message, history)
            add_to_history(channel_id, "assistant", answer)
            code_blocks = extract_code_blocks(answer)
            await interaction.followup.send(answer)
            if code_blocks:
                await send_files(interaction.channel, code_blocks)
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
            vc_data.append(f"**{vc.name}** ({len(members)}): {names}")
    if not vc_data:
        return "No one is in any voice channel."
    return "\n".join(vc_data)


@bot.tree.command(name="voice", description="See who's in voice channels")
async def slash_voice(interaction: discord.Interaction):
    if not interaction.guild:
        await interaction.response.send_message(":x: Server only.", ephemeral=True)
        return
    info = get_voice_info(interaction.guild)
    await interaction.response.send_message(f":loud_sound: **Voice channels:**\n{info}")

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
            await player.play(player.queue.get(), volume=100)
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


bot.run(TOKEN)
