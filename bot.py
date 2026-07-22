import asyncio
import os
import re
import tempfile
import textwrap
from collections import deque

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI
import wavelink
from duckduckgo_search import DDGS

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
CLOUDFLARE_KEY = os.getenv("CLOUDFLARE_API_KEY")
CLOUDFLARE_ACCOUNT = os.getenv("CLOUDFLARE_ACCOUNT_ID")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
AI_KEY = OPENROUTER_KEY or HF_TOKEN or CLOUDFLARE_KEY or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("AI_MODEL", "google/gemma-4-31b-it:free")
if MODEL.startswith("AI_MODEL="):
    MODEL = MODEL[len("AI_MODEL="):]

if not TOKEN or not AI_KEY:
    raise RuntimeError("Missing DISCORD_TOKEN or AI API key in .env file")

if OPENROUTER_KEY:
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


async def web_search(query: str, max_results: int = 5) -> str:
    try:
        results = await asyncio.to_thread(
            lambda: list(DDGS().text(query, max_results=max_results))
        )
        if not results:
            return ""
        lines = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            snippet = r.get("body", "")
            link = r.get("href", "")
            lines.append(f"[{i}] {title}\n{snippet}\n{link}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Search error: {e}"


SEARCH_TRIGGER_WORDS = {
    "latest", "current", "today", "now", "news", "weather", "price", "prices",
    "score", "scores", "update", "recent", "happened", "happening", "live",
    "stock", "crypto", "bitcoin", "election", "who won", "who won",
    "release date", "when did", "how old is", "age of", "net worth",
    "2025", "2026", "2027",
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
- Generate ANY file: /lua, /script, /file, or just say "make me a .file type..."
- Create .exe source with compile instructions
- Read & summarize channels: /read #channel
- See who's in voice: /voice or ask "who's in vc?"
- Kick/vkick/say/clear: admin commands
- Normal chat, coding help, answering questions
- See images: attach an image and ask about it
- Voice features coming soon (speech-to-text and text-to-speech)
- I remember the last 50 messages in each channel

I am a real Discord bot with real features. Never say "I can't" without checking my actual capabilities above.
"""

CHAT_SYSTEM = textwrap.dedent(f"""\
You are Null, a friendly and capable Discord bot. You are chill, concise, and cool.
{CHAT_CAPABILITIES}

Rules:
- When someone asks if you can do something, check your capabilities FIRST before saying no.
- Guide users to use the right /command for what they want.
- For code/file requests, output the content and the system sends it as a downloadable file.
- Keep responses short and natural. Don't list all features unless asked.
- Respond in the same language the user speaks.
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
        search_results = await web_search(search_query, max_results=5)
        if search_results:
            if search_results.startswith("Search error:"):
                enhanced_prompt = f"{prompt}\n\n[Web search failed: {search_results}]"
            else:
                enhanced_prompt = (
                    f"{prompt}\n\n[Web search results for '{search_query}':\n"
                    f"{search_results}\n\n"
                    f"Use the above search results to answer if helpful."
                )
            return await call_ai(CHAT_SYSTEM, enhanced_prompt, history, temperature, image_urls=image_urls)
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
        for att in message.attachments:
            if att.content_type and att.content_type.startswith("image/"):
                image_urls.append(att.url)

        channel_id = message.channel.id
        display_name = message.author.display_name
        history_entry = f"{display_name}: {content}"
        if image_urls:
            history_entry += f" [attached {len(image_urls)} image(s)]"
        add_to_history(channel_id, "user", history_entry)

        async with message.channel.typing():
            try:
                if is_create_request(content):
                    await handle_create_request(message.channel, content, reply_target=message)
                else:
                    context_extra = ""
                    content_lower = content.lower()
                    if message.guild and any(w in content_lower for w in ["voice", "vc", "vchat", "voice chat", "talk"]):
                        context_extra = f"\n\n[Server voice info: {get_voice_info(message.guild)}]"

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

    await bot.process_commands(message)


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
