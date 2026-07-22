import asyncio
import os
import re
import tempfile
import textwrap

import discord
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.5-flash-lite:free")

if not TOKEN or not OPENROUTER_KEY:
    raise RuntimeError("Missing DISCORD_TOKEN or OPENROUTER_API_KEY in .env file")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_KEY,
    default_headers={
        "HTTP-Referer": "http://localhost",
        "X-Title": "Discord Code Bot",
    },
)

EXTENSIONS = {
    "lua": ".lua",
    "luau": ".luau",
    "python": ".py",
    "py": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "ts": ".ts",
    "html": ".html",
    "css": ".css",
    "c": ".c",
    "cpp": ".cpp",
    "c++": ".cpp",
    "csharp": ".cs",
    "cs": ".cs",
    "java": ".java",
    "go": ".go",
    "rust": ".rs",
    "ruby": ".rb",
    "php": ".php",
    "swift": ".swift",
    "kotlin": ".kt",
    "sql": ".sql",
    "shell": ".sh",
    "bash": ".sh",
    "powershell": ".ps1",
    "ps1": ".ps1",
    "json": ".json",
    "yaml": ".yml",
    "xml": ".xml",
    "markdown": ".md",
    "md": ".md",
}


def extract_code_blocks(text: str) -> list[tuple[str | None, str]]:
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang.strip() if lang.strip() else None, code.strip()) for lang, code in matches]


def detect_language(text: str) -> str | None:
    text_lower = text.lower()
    if "lua" in text_lower or "luau" in text_lower or "roblox" in text_lower:
        return "lua"
    for lang in EXTENSIONS:
        if lang in text_lower:
            return lang
    return None


SYSTEM_PROMPT = textwrap.dedent("""\
You are a coding assistant bot in a Discord server. A user will ask you to write code.
Follow these rules:

1. Output ONLY the code inside a single code block with the language specified. Example:
```lua
-- code here
```

2. Do NOT include explanations, summaries, or commentary outside the code block.
3. If the user asks a non-coding question, answer briefly.
4. Always use proper syntax and follow best practices for the language.
5. If you don't know, say so honestly.
""")


intents = discord.Intents.default()
intents.message_content = True
intents.members = False
intents.presences = False
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    activity = discord.Activity(
        type=discord.ActivityType.playing,
        name="with Lua scripts",
    )
    await bot.change_presence(
        activity=activity,
        status=discord.Status.online,
    )
    print(f"Online as {bot.user}")


def _generate_code_sync(prompt: str, language: str | None) -> str:
    lang_hint = f" in {language}" if language else ""
    full_prompt = f"Write code{lang_hint} for the following request. Output ONLY the code in a code block. No explanations.\n\n{prompt}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": full_prompt},
        ],
        temperature=0.3,
        max_tokens=4096,
    )

    return response.choices[0].message.content or ""


async def generate_code(prompt: str, language: str | None) -> str:
    return await asyncio.to_thread(_generate_code_sync, prompt, language)


def choose_filename(lang: str | None, content: str) -> str:
    ext = EXTENSIONS.get(lang, ".txt") if lang else ".txt"
    first_line = content.strip().split("\n")[0] if content else "script"
    safe = re.sub(r"[^\w\-]", "", first_line[:20]) or "script"
    return f"{safe}{ext}"


@bot.command(name="script")
async def script(ctx: commands.Context, *, args: str):
    """Generate code and send as a downloadable file. Usage: !script <language> <description>"""
    parts = args.strip().split(maxsplit=1)
    lang = None
    description = args.strip()

    if len(parts) >= 2 and parts[0].lower() in EXTENSIONS:
        lang = parts[0].lower()
        description = parts[1]
    else:
        lang = detect_language(description)

    lang_display = lang or "code"
    await ctx.send(f":white_check_mark: Generating {lang_display}...")

    async with ctx.typing():
        try:
            response_text = await generate_code(description, lang)
        except Exception as e:
            await ctx.send(f":x: API error: {e}")
            return

    code_blocks = extract_code_blocks(response_text)

    if code_blocks:
        for i, (block_lang, code) in enumerate(code_blocks):
            final_lang = block_lang or lang
            filename = choose_filename(final_lang, code)

            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8"
            )
            tmp.write(code)
            tmp.close()

            ext = EXTENSIONS.get(final_lang, ".txt") if final_lang else ".txt"
            await ctx.send(
                f":package: **File {i + 1}:** `{filename}`",
                file=discord.File(tmp.name, filename=filename),
            )
            os.unlink(tmp.name)

        return

    lang_for_filename = lang or detect_language(response_text)
    filename = choose_filename(lang_for_filename, response_text)

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8"
    )
    tmp.write(response_text)
    tmp.close()

    await ctx.send(
        f":package: `{filename}`",
        file=discord.File(tmp.name, filename=filename),
    )
    os.unlink(tmp.name)


@bot.command(name="lua")
async def lua(ctx: commands.Context, *, description: str):
    """Generate a Lua script and send as a downloadable file. Usage: !lua <description>"""
    await script(ctx, args=f"lua {description}")


def _ask_sync(question: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer concisely."},
            {"role": "user", "content": question},
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    return response.choices[0].message.content or "No response."


@bot.command(name="ask")
async def ask(ctx: commands.Context, *, question: str):
    """Ask the AI a general question. Usage: !ask <question>"""
    async with ctx.typing():
        try:
            answer = await asyncio.to_thread(_ask_sync, question)
        except Exception as e:
            await ctx.send(f":x: Error: {e}")
            return

    if len(answer) > 1900:
        parts = [answer[i:i + 1900] for i in range(0, len(answer), 1900)]
        for part in parts:
            await ctx.send(part)
    else:
        await ctx.send(answer)


bot.run(TOKEN)
