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

EXTENSIONS = {
    "lua": ".lua", "luau": ".luau", "python": ".py", "py": ".py",
    "javascript": ".js", "js": ".js", "typescript": ".ts", "ts": ".ts",
    "html": ".html", "css": ".css", "c": ".c", "cpp": ".cpp", "c++": ".cpp",
    "csharp": ".cs", "cs": ".cs", "java": ".java", "go": ".go", "rust": ".rs",
    "ruby": ".rb", "php": ".php", "swift": ".swift", "kotlin": ".kt",
    "sql": ".sql", "shell": ".sh", "bash": ".sh", "powershell": ".ps1",
    "ps1": ".ps1", "json": ".json", "yaml": ".yml", "xml": ".xml",
    "markdown": ".md", "md": ".md",
}

CODE_KEYWORDS = [
    "make", "create", "write", "code", "script", "generate", "build",
    "lua", "luau", "roblox", "script", "program", "function", "class",
]

CHAT_SYSTEM = textwrap.dedent("""\
You are Null, a friendly and helpful coding assistant in a Discord server.
Your personality is chill, concise, and slightly cool.

Rules:
1. If the user asks you to write code, output the code in a code block with the language tag. Include a brief one-line explanation before the block.
2. If the user is just chatting, reply conversationally. Keep it short.
3. Always use proper syntax and follow best practices.
4. If you don't know something, be honest.
5. Respond in the same language the user speaks.
""")

CODE_SYSTEM = textwrap.dedent("""\
You are a coding assistant. Write code based on the user's request.
Output ONLY the code in a code block. No explanations.
Follow best practices and proper syntax.
""")


def extract_code_blocks(text: str) -> list[tuple[str | None, str]]:
    pattern = r"```(\w*)\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return [(lang.strip() if lang.strip() else None, code.strip()) for lang, code in matches]


def detect_language(text: str) -> str | None:
    text_lower = text.lower()
    if any(w in text_lower for w in ["lua", "luau", "roblox"]):
        return "lua"
    for lang in EXTENSIONS:
        if lang in text_lower:
            return lang
    return None


def is_code_request(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in CODE_KEYWORDS)


def choose_filename(lang: str | None, content: str) -> str:
    ext = EXTENSIONS.get(lang, ".txt") if lang else ".txt"
    first_line = content.strip().split("\n")[0] if content else "script"
    safe = re.sub(r"[^\w\-]", "", first_line[:20]) or "script"
    return f"{safe}{ext}"


async def send_code_files(channel, code_blocks, lang_hint=None):
    for i, (block_lang, code) in enumerate(code_blocks):
        final_lang = block_lang or lang_hint
        filename = choose_filename(final_lang, code)
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
                if is_code_request(content):
                    lang = detect_language(content)
                    text = await call_ai(CODE_SYSTEM, content, temperature=0.2)
                    code_blocks = extract_code_blocks(text)
                    if code_blocks:
                        await send_code_files(message.channel, code_blocks, lang)
                    else:
                        filename = choose_filename(lang, text)
                        tmp = tempfile.NamedTemporaryFile(
                            mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8"
                        )
                        tmp.write(text)
                        tmp.close()
                        await message.reply(
                            f":package: `{filename}`",
                            file=discord.File(tmp.name, filename=filename),
                        )
                        os.unlink(tmp.name)
                else:
                    answer = await call_ai(CHAT_SYSTEM, content, temperature=0.7)
                    code_blocks = extract_code_blocks(answer)
                    await message.reply(answer, mention_author=False)
                    if code_blocks:
                        await send_code_files(message.channel, code_blocks)
            except Exception as e:
                await message.reply(f":x: Error: {e}", mention_author=False)
        return

    await bot.process_commands(message)


@bot.tree.command(name="hey", description="Chat freely with Null")
async def slash_hey(interaction: discord.Interaction, message: str):
    await interaction.response.defer()
    try:
        if is_code_request(message):
            lang = detect_language(message)
            text = await call_ai(CODE_SYSTEM, message, temperature=0.2)
            code_blocks = extract_code_blocks(text)
            if code_blocks:
                await interaction.followup.send(
                    ":white_check_mark: Here's your code:"
                )
                await send_code_files(interaction.channel, code_blocks, lang)
            else:
                filename = choose_filename(lang, text)
                tmp = tempfile.NamedTemporaryFile(
                    mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8"
                )
                tmp.write(text)
                tmp.close()
                await interaction.followup.send(
                    f":package: `{filename}`",
                    file=discord.File(tmp.name, filename=filename),
                )
                os.unlink(tmp.name)
        else:
            answer = await call_ai(CHAT_SYSTEM, message, temperature=0.7)
            code_blocks = extract_code_blocks(answer)
            await interaction.followup.send(answer)
            if code_blocks:
                await send_code_files(interaction.channel, code_blocks)
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
            await send_code_files(interaction.channel, code_blocks, "lua")
        else:
            filename = choose_filename("lua", text)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8"
            )
            tmp.write(text)
            tmp.close()
            await interaction.followup.send(
                f":package: `{filename}`",
                file=discord.File(tmp.name, filename=filename),
            )
            os.unlink(tmp.name)
    except Exception as e:
        await interaction.followup.send(f":x: Error: {e}")


@bot.tree.command(name="script", description="Generate code in any language")
async def slash_script(interaction: discord.Interaction, language: str, description: str):
    await interaction.response.defer()
    lang = language.lower()
    if lang not in EXTENSIONS:
        await interaction.followup.send(f":x: Unknown language. Try: lua, python, js, etc.")
        return
    try:
        prompt = f"Write a {language} script that: {description}"
        text = await call_ai(CODE_SYSTEM, prompt, temperature=0.2)
        code_blocks = extract_code_blocks(text)
        if code_blocks:
            await interaction.followup.send(f":white_check_mark: Here's your {language} code:")
            await send_code_files(interaction.channel, code_blocks, lang)
        else:
            filename = choose_filename(lang, text)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=f"_{filename}", delete=False, encoding="utf-8"
            )
            tmp.write(text)
            tmp.close()
            await interaction.followup.send(
                f":package: `{filename}`",
                file=discord.File(tmp.name, filename=filename),
            )
            os.unlink(tmp.name)
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
            await send_code_files(interaction.channel, code_blocks)
    except Exception as e:
        await interaction.followup.send(f":x: Error: {e}")


bot.run(TOKEN)
