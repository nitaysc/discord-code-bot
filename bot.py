import asyncio
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from collections import deque

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv
from openai import OpenAI
import yt_dlp

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not TOKEN or not GEMINI_KEY:
    raise RuntimeError("Missing DISCORD_TOKEN or GEMINI_API_KEY in .env file")

client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=GEMINI_KEY,
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
- Play music from YouTube: use /play <song name>. Also /skip /stop /queue /pause /resume.
- Generate ANY file: /lua, /script, /file, or just say "make me a .file type..."
- Create .exe source with compile instructions
- Read & summarize channels: /read #channel
- See who's in voice: /voice or ask "who's in vc?"
- Kick/vkick/say/clear: admin commands
- Normal chat, coding help, answering questions
- See images: attach an image and ask about it
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
                    answer = await call_ai(CHAT_SYSTEM, prompt, history, temperature=0.7,
                                          image_urls=image_urls if image_urls else None)
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
            answer = await call_ai(CHAT_SYSTEM, message, history, temperature=0.7)
            add_to_history(channel_id, "assistant", answer)
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


YTDL_OPTS = {
    "format": "bestaudio/best",
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "default_search": "ytsearch",
    "extract_flat": "in_playlist",
}

YTDL_STREAM_OPTS = {
    "format": "bestaudio/best",
    "quiet": True,
    "no_warnings": True,
    "output": "-",
    "default_search": "ytsearch",
}

FFMPEG_OPTS = {
    "before_options": "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5",
    "options": "-vn",
}

music_queues: dict[int, deque] = {}
music_current: dict[int, dict] = {}


def get_music_queue(guild_id: int) -> deque:
    if guild_id not in music_queues:
        music_queues[guild_id] = deque()
    return music_queues[guild_id]


async def play_next(guild: discord.Guild, voice_client: discord.VoiceClient):
    queue = get_music_queue(guild.id)
    if not queue:
        music_current.pop(guild.id, None)
        await asyncio.sleep(120)
        if guild.id not in music_current and voice_client.is_connected():
            await voice_client.disconnect()
        return

    song = queue.popleft()
    music_current[guild.id] = song

    try:
        ytdl_cmd = [
            sys.executable, "-m", "yt_dlp", song["url"],
            "-f", "bestaudio/best",
            "-o", "-",
            "-q", "--no-warnings",
            "--default-search", "ytsearch",
            "--audio-format", "mp3",
        ]
        proc = subprocess.Popen(ytdl_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        source = discord.PCMVolumeTransformer(
            discord.FFmpegPCMAudio(
                proc.stdout,
                pipe=True,
                before_options="-f mp3"
            )
        )
        voice_client.play(source, after=lambda e: asyncio.run_coroutine_threadsafe(
            play_next(guild, voice_client), bot.loop
        ))
    except Exception as e:
        print(f"[MUSIC] Error: {e}")
        await asyncio.sleep(0.5)
        await play_next(guild, voice_client)


@bot.tree.command(name="play", description="Play music from YouTube")
@app_commands.describe(query="Song name or YouTube URL")
async def slash_play(interaction: discord.Interaction, query: str):
    if not interaction.user.voice:
        await interaction.response.send_message(":x: Join a voice channel first!", ephemeral=True)
        return

    await interaction.response.defer()
    voice = interaction.user.voice.channel
    voice_client = interaction.guild.voice_client

    if not voice_client:
        voice_client = await voice.connect()
    elif voice_client.channel != voice:
        await voice_client.move_to(voice)

    try:
        with yt_dlp.YoutubeDL(YTDL_OPTS) as ydl:
            info = ydl.extract_info(query, download=False)
            if "entries" in info:
                info = info["entries"][0]
            title = info.get("title", "Unknown")
            url = info.get("webpage_url", query)

        queue = get_music_queue(interaction.guild_id)
        song = {"title": title, "url": url, "requester": interaction.user.display_name}
        queue.append(song)

        if not voice_client.is_playing() and interaction.guild_id not in music_current:
            await play_next(interaction.guild, voice_client)
            await interaction.followup.send(f":musical_note: Now playing: **{title}**")
        else:
            await interaction.followup.send(f":notes: Queued (#{len(queue)}): **{title}**")
    except Exception as e:
        await interaction.followup.send(f":x: Failed: {e}")


@bot.tree.command(name="skip", description="Skip current song")
async def slash_skip(interaction: discord.Interaction):
    voice_client = interaction.guild.voice_client
    if not voice_client or not voice_client.is_playing():
        await interaction.response.send_message(":x: Nothing playing.", ephemeral=True)
        return
    voice_client.stop()
    await interaction.response.send_message(":track_next: Skipped.")


@bot.tree.command(name="stop", description="Stop music and leave voice")
async def slash_stop(interaction: discord.Interaction):
    voice_client = interaction.guild.voice_client
    if voice_client:
        music_queues.pop(interaction.guild_id, None)
        music_current.pop(interaction.guild_id, None)
        voice_client.stop()
        await voice_client.disconnect()
    await interaction.response.send_message(":stop_button: Stopped.")


@bot.tree.command(name="queue", description="Show music queue")
async def slash_queue(interaction: discord.Interaction):
    queue = get_music_queue(interaction.guild_id)
    current = music_current.get(interaction.guild_id)
    lines = []
    if current:
        lines.append(f":musical_note: **Now**: {current['title']}")
    if queue:
        for i, song in enumerate(queue, 1):
            lines.append(f"  #{i} {song['title']}")
    if not lines:
        await interaction.response.send_message(":x: Queue is empty.", ephemeral=True)
    else:
        await interaction.response.send_message("\n".join(lines))


@bot.tree.command(name="pause", description="Pause music")
async def slash_pause(interaction: discord.Interaction):
    vc = interaction.guild.voice_client
    if vc and vc.is_playing():
        vc.pause()
        await interaction.response.send_message(":pause_button: Paused.")
    else:
        await interaction.response.send_message(":x: Nothing to pause.", ephemeral=True)


@bot.tree.command(name="resume", description="Resume music")
async def slash_resume(interaction: discord.Interaction):
    vc = interaction.guild.voice_client
    if vc and vc.is_paused():
        vc.resume()
        await interaction.response.send_message(":arrow_forward: Resumed.")
    else:
        await interaction.response.send_message(":x: Nothing to resume.", ephemeral=True)


bot.run(TOKEN)
