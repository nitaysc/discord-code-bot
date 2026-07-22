import asyncio
import os
import re
import tempfile
import textwrap

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")

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

CHAT_SYSTEM = textwrap.dedent("""\
You are Null, a friendly coding assistant in a Discord server. You are chill, concise, and cool.

You can generate ANY text-based file: code, scripts, configs, data files, documents, etc.
The system will automatically save your output as a downloadable file.

Rules:
- If asked to create/write/make something, output the content raw (no code blocks needed for non-code files).
- For CODE files, wrap in a code block with the language tag (```python, ```lua, etc).
- For non-code text files (json, xml, yaml, csv, bat, txt, configs, etc), output raw content.
- If you CAN'T create something (like .exe, images, PDFs), explain why and offer alternatives.
- For general chat, reply conversationally and keep it short.
- Follow best practices for whatever format you're generating.
""")

CODE_SYSTEM = textwrap.dedent("""\
You are a coding assistant. Write code based on the user's request.
Output ONLY the code in a code block with language tag. No explanations.
Follow best practices and proper syntax.
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
    return any(kw in text_lower for kw in CREATE_KEYWORDS)


def choose_filename(file_type: str | None, content: str, index: int = 0) -> str:
    ext = FILE_EXTENSIONS.get(file_type, ".txt") if file_type else ".txt"
    first_line = content.strip().split("\n")[0] if content else "output"
    safe = re.sub(r"[^\w\-]", "_", first_line[:25]) or "output"
    if index > 0:
        safe = f"{safe}_{index}"
    return f"{safe}{ext}"


async def send_files(channel, code_blocks, lang_hint=None):
    for i, (block_lang, code) in enumerate(code_blocks):
        final_type = block_lang or lang_hint
        filename = choose_filename(final_type, code, i)
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8"
        )
        tmp.write(code)
        tmp.close()
        await channel.send(
            f":package: `{filename}`",
            file=discord.File(tmp.name, filename=filename),
        )
        os.unlink(tmp.name)


async def send_raw_file(channel, content: str, file_type: str | None):
    filename = choose_filename(file_type, content)
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8"
    )
    tmp.write(content)
    tmp.close()
    await channel.send(
        f":package: `{filename}`",
        file=discord.File(tmp.name, filename=filename),
    )
    os.unlink(tmp.name)


def _call_ai(system: str, prompt: str, temperature: float = 0.5, max_tokens: int = 4096) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


async def call_ai(system: str, prompt: str, temperature: float = 0.5, max_tokens: int = 4096) -> str:
    return await asyncio.to_thread(_call_ai, system, prompt, temperature, max_tokens)


async def handle_create_request(channel, prompt: str, reply_target=None):
    file_type, file_kind = detect_file_type(prompt)

    if file_kind == "binary":
        msg = (
            f":x: I can't create `.{
                file_type}` files — that's a binary format.\n"
            f"Try asking for source code instead! For example:\n"
            f"- `make me a .bat file that opens a message`\n"
            f"- `write a Python script that does X`\n"
            f"- `create a PowerShell .ps1 script`"
        )
        if reply_target:
            await reply_target.reply(msg, mention_author=False)
        else:
            await channel.send(msg)
        return

    is_code_type = file_type in LANG_EXTENSIONS or file_type is None
    system = CODE_SYSTEM if is_code_type else CHAT_SYSTEM
    text = await call_ai(system, prompt, temperature=0.2)

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

    await send_raw_file(channel, text, file_type)


intents = discord.Intents.default()
intents.message_content = True
intents.members = False
intents.presences = False


class CodeBot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix="!", intents=intents)

    async def setup_hook(self):
        await self.tree.sync()

    async def on_ready(self):
        activity = discord.Activity(
            type=discord.ActivityType.watching,
            name="for /lua or @Null",
        )
        await self.change_presence(activity=activity, status=discord.Status.online)
        print(f"Online as {self.user}")


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

        async with message.channel.typing():
            try:
                if is_create_request(content):
                    await handle_create_request(message.channel, content, reply_target=message)
                else:
                    answer = await call_ai(CHAT_SYSTEM, content, temperature=0.7)
                    code_blocks = extract_code_blocks(answer)
                    await message.reply(answer, mention_author=False)
                    if code_blocks:
                        await send_files(message.channel, code_blocks)
            except Exception as e:
                await message.reply(f":x: Error: {e}", mention_author=False)
        return

    await bot.process_commands(message)


@bot.tree.command(name="hey", description="Chat with Null or ask it to create files")
async def slash_hey(interaction: discord.Interaction, message: str):
    await interaction.response.defer()
    try:
        if is_create_request(message):
            await handle_create_request(interaction.channel, message)
        else:
            answer = await call_ai(CHAT_SYSTEM, message, temperature=0.7)
            code_blocks = extract_code_blocks(answer)
            await interaction.followup.send(answer)
            if code_blocks:
                await send_files(interaction.channel, code_blocks)
    except Exception as e:
        await interaction.followup.send(f":x: Error: {e}")


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
        await interaction.followup.send(
            f":x: `.{
                ftype}` is a binary format — I can only create text-based files.\n"
            f"Try: bat, ps1, py, js, html, json, csv, txt, xml, ini, reg, etc."
        )
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
    try:
        answer = await call_ai(CHAT_SYSTEM, question, temperature=0.7)
        code_blocks = extract_code_blocks(answer)
        await interaction.followup.send(answer)
        if code_blocks:
            await send_files(interaction.channel, code_blocks)
    except Exception as e:
        await interaction.followup.send(f":x: Error: {e}")


bot.run(TOKEN)
