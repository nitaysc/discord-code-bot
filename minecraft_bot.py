import asyncio
import uuid
import re
from typing import Callable, Optional

HAS_MC_BOT = False
try:
    from pymcbotlite import Bot, BotError
    HAS_MC_BOT = True
except ImportError:
    pass


class NoopWorld:
    """Discards all chunk data to save memory."""
    registry = None
    min_y = -64
    view_chunk_x = 0
    view_chunk_z = 0
    def known_block_count(self): return 0
    def known_chunk_count(self): return 0
    def set_block(self, *a, **kw): pass
    def get_block(self, *a, **kw): return None
    def unload_chunk(self, *a, **kw): pass
    def load_chunk_data(self, *a, **kw): return 0
    def snapshot(self): return self


class MCBotManager:
    def __init__(self):
        self.bot: Optional[Bot] = None
        self._connected = False
        self._follow_task: Optional[asyncio.Task] = None
        self._follow_target: Optional[str] = None
        self._chat_callback: Optional[Callable] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._mc_channel_id: Optional[int] = None
        self.server_host = ""
        self.server_port = 25565
        self.bot_username = "NullBot"
        self._events_registered = False

    def _init_bot(self):
        """Pre-create the Bot instance at startup to spread out memory cost."""
        if HAS_MC_BOT and not self.bot:
            self.bot = Bot(host="localhost", port=25565, account=None, auto_reconnect=True, send_client_ticks=False, physics=False)
            self.bot.client.account = "mcbot"
            self.bot.client.world = NoopWorld()

    @property
    def connected(self) -> bool:
        return self._connected and self.bot is not None

    async def connect(self, host: str, port: int = 25565, username: str = "NullBot",
                      chat_callback: Optional[Callable] = None,
                      mc_channel_id: Optional[int] = None) -> str:
        if not HAS_MC_BOT:
            return ":x: pymcbotlite not installed. Run: pip install pymcbotlite"
        if self._connected:
            await self.disconnect()
        self.server_host = host
        self.server_port = port
        self.bot_username = username
        self._chat_callback = chat_callback
        self._mc_channel_id = mc_channel_id
        self._loop = asyncio.get_running_loop()
        try:
            if not self.bot:
                self._init_bot()
            if not self._events_registered:
                self._events_registered = True
                @self.bot.event
                async def on_chat(message):
                    if self._chat_callback:
                        sender = message.sender or "Server"
                        await self._chat_callback(sender, message.clean)

                @self.bot.event
                async def on_ready():
                    print(f"[MC BOT] Connected as {self.bot.username} to {self.server_host}:{self.server_port}")

                @self.bot.event
                async def on_disconnect(reason):
                    print(f"[MC BOT] Disconnected: {reason}")
                    self._connected = False

                @self.bot.event
                async def on_error(exc):
                    print(f"[MC BOT] Play loop error: {exc}")
            self.bot.client.host = host
            self.bot.client.port = port
            self.bot.client.username = username
            self.bot.client.uuid = str(uuid.uuid4())

            await self.bot.client.connect()
            self._connected = True
            return f":white_check_mark: Joined **{host}:{port}** as **{username}**"
        except Exception as e:
            self._connected = False
            self.bot = None
            return f":x: Failed to connect: {e}"

    async def disconnect(self):
        self._stop_follow()
        if self.bot:
            try:
                await self.bot.disconnect()
            except Exception:
                pass
        self._connected = False
        self.bot = None

    async def send_chat(self, message: str):
        if self.bot and self._connected:
            try:
                await self.bot.chat(message[:256])
            except Exception as e:
                print(f"[MC BOT] chat error: {e}")

    async def follow(self, target: str):
        if not self.bot or not self._connected:
            return ":x: Not connected to a Minecraft server"
        self._stop_follow()
        self._follow_target = target
        self._follow_task = asyncio.create_task(self._follow_loop(target))
        return f":eyes: Following **{target}**"

    async def _follow_loop(self, target: str):
        while self._connected and self._follow_target == target:
            try:
                players = self.bot.entities
                target_id = None
                target_pos = None
                for eid, data in players.items():
                    name = data.get("name", "")
                    if name and target.lower() in name.lower():
                        target_id = eid
                        target_pos = (data.get("x", 0), data.get("y", 0), data.get("z", 0))
                        break
                if target_pos:
                    x, y, z = target_pos
                    dx = self.bot.client.x - x
                    dz = self.bot.client.z - z
                    dist = (dx * dx + dz * dz) ** 0.5
                    if dist > 2.5:
                        await self.bot.move_path_to(x, y, z, stop_distance=2)
                    if dist < 10:
                        await self.bot.look_at(x, y + 1.5, z)
                    await asyncio.sleep(0.3)
                else:
                    await asyncio.sleep(1)
            except Exception as e:
                print(f"[MC BOT] follow error: {e}")
                await asyncio.sleep(1)

    def _stop_follow(self):
        self._follow_target = None
        if self._follow_task:
            self._follow_task.cancel()
            self._follow_task = None

    async def stop(self):
        self._stop_follow()
        return ":stop_button: Stopped all actions"

    async def come_to(self, x: float, y: float, z: float):
        if not self.bot or not self._connected:
            return ":x: Not connected to a Minecraft server"
        try:
            self._stop_follow()
            await self.bot.move_path_to(x, y, z, stop_distance=1)
            return ":arrow_forward: Moving to your position"
        except Exception as e:
            return f":x: Movement failed: {e}"

    async def goto(self, x: float, y: float, z: float):
        if not self.bot or not self._connected:
            return ":x: Not connected to a Minecraft server"
        try:
            self._stop_follow()
            await self.bot.move_path_to(x, y, z, stop_distance=1)
            return f":arrow_forward: Moving to {x:.0f}, {y:.0f}, {z:.0f}"
        except Exception as e:
            return f":x: Movement failed: {e}"

    async def look_at(self, x: float, y: float, z: float):
        if not self.bot or not self._connected:
            return ":x: Not connected to a Minecraft server"
        try:
            await self.bot.look_at(x, y, z)
            return ":eyes: Looked at target"
        except Exception as e:
            return f":x: Failed: {e}"

    async def attack(self, target: str):
        if not self.bot or not self._connected:
            return ":x: Not connected to a Minecraft server"
        try:
            players = self.bot.entities
            for eid, data in players.items():
                name = data.get("name", "")
                if name and target.lower() in name.lower():
                    await self.bot.attack(eid)
                    return f":crossed_swords: Attacked **{name}**"
            return f":x: Could not find **{target}** nearby"
        except Exception as e:
            return f":x: Attack failed: {e}"

    async def mine(self):
        if not self.bot or not self._connected:
            return ":x: Not connected to a Minecraft server"
        try:
            import math
            yaw = self.bot.client.yaw
            pitch = self.bot.client.pitch
            x = self.bot.client.x + math.sin(math.radians(yaw)) * math.cos(math.radians(pitch)) * 4.5
            z = self.bot.client.z - math.cos(math.radians(yaw)) * math.cos(math.radians(pitch)) * 4.5
            y = self.bot.client.y + 1.6 - math.sin(math.radians(pitch)) * 4.5
            await self.bot.mine_block(int(x), int(y), int(z))
            return ":pick: Mining block"
        except Exception as e:
            return f":x: Mining failed: {e}"

    def get_position(self) -> Optional[tuple]:
        if self.bot and self._connected:
            return (self.bot.client.x, self.bot.client.y, self.bot.client.z)
        return None
