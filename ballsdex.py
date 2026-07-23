import asyncio
import io
import math
import random
import time
from datetime import datetime
from typing import Optional

import discord
from discord import app_commands
from PIL import Image, ImageDraw, ImageFont

# ─── Countryball Data ───────────────────────────────────────────────
# All 195+ countries with emoji flags, rarity tiers, and base stats
COUNTRY_DATA = [
    # Legendary (rarest)
    ("United States", "US", "\U0001f1fa\U0001f1f8", "Legendary", 80, 70, 85, 75),
    ("China", "CN", "\U0001f1e8\U0001f1f3", "Legendary", 75, 80, 90, 70),
    ("Russia", "RU", "\U0001f1f7\U0001f1fa", "Legendary", 85, 85, 85, 60),
    ("Germany", "DE", "\U0001f1e9\U0001f1ea", "Legendary", 80, 75, 75, 80),
    ("United Kingdom", "GB", "\U0001f1ec\U0001f1e7", "Legendary", 70, 75, 80, 85),
    ("Japan", "JP", "\U0001f1ef\U0001f1f5", "Legendary", 75, 70, 70, 95),
    ("France", "FR", "\U0001f1eb\U0001f1f7", "Legendary", 70, 65, 75, 85),
    ("India", "IN", "\U0001f1ee\U0001f1f3", "Legendary", 75, 70, 85, 75),
    ("Brazil", "BR", "\U0001f1e7\U0001f1f7", "Legendary", 80, 65, 80, 80),
    ("Canada", "CA", "\U0001f1e8\U0001f1e6", "Legendary", 65, 75, 85, 70),
    # Epic
    ("Australia", "AU", "\U0001f1e6\U0001f1fa", "Epic", 70, 65, 75, 80),
    ("Italy", "IT", "\U0001f1ee\U0001f1f9", "Epic", 60, 60, 65, 85),
    ("Spain", "ES", "\U0001f1ea\U0001f1f8", "Epic", 65, 60, 70, 80),
    ("South Korea", "KR", "\U0001f1f0\U0001f1f7", "Epic", 70, 65, 65, 90),
    ("Israel", "IL", "\U0001f1ee\U0001f1f1", "Epic", 75, 70, 60, 85),
    ("Turkey", "TR", "\U0001f1f9\U0001f1f7", "Epic", 70, 70, 75, 65),
    ("Saudi Arabia", "SA", "\U0001f1f8\U0001f1e6", "Epic", 65, 75, 70, 60),
    ("Sweden", "SE", "\U0001f1f8\U0001f1ea", "Epic", 60, 70, 70, 75),
    ("Netherlands", "NL", "\U0001f1f3\U0001f1f1", "Epic", 60, 65, 65, 80),
    ("Switzerland", "CH", "\U0001f1e8\U0001f1ed", "Epic", 55, 80, 70, 70),
    ("Argentina", "AR", "\U0001f1e6\U0001f1f7", "Epic", 75, 60, 70, 75),
    ("Mexico", "MX", "\U0001f1f2\U0001f1fd", "Epic", 70, 65, 75, 70),
    ("Ukraine", "UA", "\U0001f1fa\U0001f1e6", "Epic", 65, 70, 75, 65),
    ("Poland", "PL", "\U0001f1f5\U0001f1f1", "Epic", 65, 70, 75, 65),
    ("Egypt", "EG", "\U0001f1ea\U0001f1ec", "Epic", 70, 65, 75, 60),
    ("Norway", "NO", "\U0001f1f3\U0001f1f4", "Epic", 65, 65, 70, 75),
    ("Greece", "GR", "\U0001f1ec\U0001f1f7", "Epic", 60, 65, 65, 75),
    ("South Africa", "ZA", "\U0001f1ff\U0001f1e6", "Epic", 70, 65, 75, 70),
    ("Thailand", "TH", "\U0001f1f9\U0001f1ed", "Epic", 65, 60, 70, 80),
    ("Nigeria", "NG", "\U0001f1f3\U0001f1ec", "Epic", 75, 65, 80, 65),
    # Rare
    ("Portugal", "PT", "\U0001f1f5\U0001f1f9", "Rare", 55, 60, 60, 70),
    ("Denmark", "DK", "\U0001f1e9\U0001f1f0", "Rare", 55, 60, 60, 70),
    ("Finland", "FI", "\U0001f1eb\U0001f1ee", "Rare", 55, 65, 65, 70),
    ("Ireland", "IE", "\U0001f1ee\U0001f1ea", "Rare", 55, 55, 60, 75),
    ("Austria", "AT", "\U0001f1e6\U0001f1f9", "Rare", 55, 65, 60, 65),
    ("Belgium", "BE", "\U0001f1e7\U0001f1ea", "Rare", 50, 65, 60, 65),
    ("Czech Republic", "CZ", "\U0001f1e8\U0001f1ff", "Rare", 60, 60, 65, 65),
    ("Hungary", "HU", "\U0001f1ed\U0001f1fa", "Rare", 65, 55, 65, 65),
    ("Romania", "RO", "\U0001f1f7\U0001f1f4", "Rare", 60, 60, 65, 60),
    ("Chile", "CL", "\U0001f1e8\U0001f1f1", "Rare", 60, 55, 65, 70),
    ("Colombia", "CO", "\U0001f1e8\U0001f1f4", "Rare", 65, 55, 65, 65),
    ("Peru", "PE", "\U0001f1f5\U0001f1ea", "Rare", 55, 55, 65, 65),
    ("Philippines", "PH", "\U0001f1f5\U0001f1ed", "Rare", 60, 55, 65, 75),
    ("Vietnam", "VN", "\U0001f1fb\U0001f1f3", "Rare", 65, 55, 70, 75),
    ("Malaysia", "MY", "\U0001f1f2\U0001f1fe", "Rare", 55, 60, 65, 65),
    ("Singapore", "SG", "\U0001f1f8\U0001f1ec", "Rare", 50, 70, 55, 85),
    ("New Zealand", "NZ", "\U0001f1f3\U0001f1ff", "Rare", 55, 55, 60, 75),
    ("Iraq", "IQ", "\U0001f1ee\U0001f1f6", "Rare", 65, 60, 70, 55),
    ("Iran", "IR", "\U0001f1ee\U0001f1f7", "Rare", 65, 65, 70, 55),
    ("Pakistan", "PK", "\U0001f1f5\U0001f1f0", "Rare", 65, 60, 75, 60),
    ("Kenya", "KE", "\U0001f1f0\U0001f1ea", "Rare", 70, 55, 75, 70),
    ("Morocco", "MA", "\U0001f1f2\U0001f1e6", "Rare", 60, 60, 65, 65),
    ("Venezuela", "VE", "\U0001f1fb\U0001f1ea", "Rare", 60, 55, 65, 60),
    ("Cuba", "CU", "\U0001f1e8\U0001f1fa", "Rare", 60, 65, 65, 55),
    ("North Korea", "KP", "\U0001f1f0\U0001f1f5", "Rare", 55, 70, 65, 50),
    # Uncommon
    ("Serbia", "RS", "\U0001f1f7\U0001f1f8", "Uncommon", 60, 55, 60, 60),
    ("Croatia", "HR", "\U0001f1ed\U0001f1f7", "Uncommon", 55, 55, 55, 65),
    ("Bulgaria", "BG", "\U0001f1e7\U0001f1ec", "Uncommon", 55, 55, 60, 55),
    ("Slovakia", "SK", "\U0001f1f8\U0001f1f0", "Uncommon", 50, 55, 55, 60),
    ("Slovenia", "SI", "\U0001f1f8\U0001f1ee", "Uncommon", 50, 55, 55, 60),
    ("Lithuania", "LT", "\U0001f1f1\U0001f1f9", "Uncommon", 55, 55, 60, 60),
    ("Latvia", "LV", "\U0001f1f1\U0001f1fb", "Uncommon", 50, 55, 55, 60),
    ("Estonia", "EE", "\U0001f1ea\U0001f1ea", "Uncommon", 50, 55, 55, 65),
    ("Algeria", "DZ", "\U0001f1e9\U0001f1ff", "Uncommon", 60, 55, 65, 55),
    ("Angola", "AO", "\U0001f1e6\U0001f1f4", "Uncommon", 60, 55, 65, 55),
    ("Bangladesh", "BD", "\U0001f1e7\U0001f1e9", "Uncommon", 55, 50, 65, 60),
    ("Belarus", "BY", "\U0001f1e7\U0001f1fe", "Uncommon", 55, 60, 65, 55),
    ("Bolivia", "BO", "\U0001f1e7\U0001f1f4", "Uncommon", 55, 55, 60, 55),
    ("Cambodia", "KH", "\U0001f1f0\U0001f1ed", "Uncommon", 50, 50, 55, 65),
    ("Costa Rica", "CR", "\U0001f1e8\U0001f1f7", "Uncommon", 45, 55, 55, 65),
    ("Dominican Republic", "DO", "\U0001f1e9\U0001f1f4", "Uncommon", 55, 50, 60, 65),
    ("Ecuador", "EC", "\U0001f1ea\U0001f1e8", "Uncommon", 55, 50, 60, 60),
    ("Ethiopia", "ET", "\U0001f1ea\U0001f1f9", "Uncommon", 55, 50, 70, 55),
    ("Ghana", "GH", "\U0001f1ec\U0001f1ed", "Uncommon", 60, 50, 65, 60),
    ("Guatemala", "GT", "\U0001f1ec\U0001f1f9", "Uncommon", 50, 50, 55, 55),
    ("Honduras", "HN", "\U0001f1ed\U0001f1f3", "Uncommon", 50, 50, 55, 55),
    ("Iceland", "IS", "\U0001f1ee\U0001f1f8", "Uncommon", 45, 55, 55, 65),
    ("Indonesia", "ID", "\U0001f1ee\U0001f1e9", "Uncommon", 55, 50, 65, 65),
    ("Jordan", "JO", "\U0001f1ef\U0001f1f4", "Uncommon", 55, 55, 60, 55),
    ("Kazakhstan", "KZ", "\U0001f1f0\U0001f1ff", "Uncommon", 55, 60, 65, 55),
    ("Kuwait", "KW", "\U0001f1f0\U0001f1fc", "Uncommon", 45, 65, 55, 60),
    ("Lebanon", "LB", "\U0001f1f1\U0001f1e7", "Uncommon", 55, 50, 55, 65),
    ("Libya", "LY", "\U0001f1f1\U0001f1fe", "Uncommon", 55, 55, 60, 55),
    ("Luxembourg", "LU", "\U0001f1f1\U0001f1fa", "Uncommon", 40, 65, 50, 65),
    ("Madagascar", "MG", "\U0001f1f2\U0001f1ec", "Uncommon", 50, 50, 60, 55),
    ("Mongolia", "MN", "\U0001f1f2\U0001f1f3", "Uncommon", 60, 60, 65, 60),
    ("Myanmar", "MM", "\U0001f1f2\U0001f1f2", "Uncommon", 55, 50, 60, 55),
    ("Nepal", "NP", "\U0001f1f3\U0001f1f5", "Uncommon", 50, 50, 55, 60),
    ("Nicaragua", "NI", "\U0001f1f3\U0001f1ee", "Uncommon", 50, 50, 55, 55),
    ("Oman", "OM", "\U0001f1f4\U0001f1f2", "Uncommon", 50, 60, 55, 55),
    ("Panama", "PA", "\U0001f1f5\U0001f1e6", "Uncommon", 45, 55, 55, 65),
    ("Paraguay", "PY", "\U0001f1f5\U0001f1fe", "Uncommon", 55, 50, 60, 60),
    ("Qatar", "QA", "\U0001f1f6\U0001f1e6", "Uncommon", 40, 65, 50, 65),
    ("Sri Lanka", "LK", "\U0001f1f1\U0001f1f0", "Uncommon", 50, 50, 55, 60),
    ("Sudan", "SD", "\U0001f1f8\U0001f1e9", "Uncommon", 55, 50, 65, 50),
    ("Syria", "SY", "\U0001f1f8\U0001f1fe", "Uncommon", 55, 55, 60, 55),
    ("Tanzania", "TZ", "\U0001f1f9\U0001f1ff", "Uncommon", 55, 50, 65, 55),
    ("Tunisia", "TN", "\U0001f1f9\U0001f1f3", "Uncommon", 55, 55, 60, 55),
    ("Uganda", "UG", "\U0001f1fa\U0001f1ec", "Uncommon", 55, 50, 65, 55),
    ("United Arab Emirates", "AE", "\U0001f1e6\U0001f1ea", "Uncommon", 45, 70, 50, 70),
    ("Uruguay", "UY", "\U0001f1fa\U0001f1fe", "Uncommon", 55, 55, 60, 65),
    ("Uzbekistan", "UZ", "\U0001f1fa\U0001f1ff", "Uncommon", 55, 55, 60, 55),
    ("Yemen", "YE", "\U0001f1fe\U0001f1ea", "Uncommon", 50, 50, 55, 50),
    ("Zimbabwe", "ZW", "\U0001f1ff\U0001f1fc", "Uncommon", 55, 50, 60, 55),
    # Common (many smaller countries)
    ("Albania", "AL", "\U0001f1e6\U0001f1f1", "Common", 50, 45, 50, 55),
    ("Andorra", "AD", "\U0001f1e6\U0001f1e9", "Common", 35, 50, 40, 60),
    ("Armenia", "AM", "\U0001f1e6\U0001f1f2", "Common", 55, 50, 55, 55),
    ("Azerbaijan", "AZ", "\U0001f1e6\U0001f1ff", "Common", 55, 50, 55, 55),
    ("Bahrain", "BH", "\U0001f1e7\U0001f1ed", "Common", 40, 55, 45, 60),
    ("Barbados", "BB", "\U0001f1e7\U0001f1e7", "Common", 40, 45, 45, 60),
    ("Bhutan", "BT", "\U0001f1e7\U0001f1f9", "Common", 45, 50, 50, 55),
    ("Bosnia", "BA", "\U0001f1e7\U0001f1e6", "Common", 55, 50, 55, 55),
    ("Brunei", "BN", "\U0001f1e7\U0001f1f3", "Common", 40, 55, 45, 55),
    ("Burkina Faso", "BF", "\U0001f1e7\U0001f1eb", "Common", 45, 45, 55, 50),
    ("Burundi", "BI", "\U0001f1e7\U0001f1ee", "Common", 45, 40, 55, 50),
    ("Cabo Verde", "CV", "\U0001f1e8\U0001f1fb", "Common", 40, 45, 45, 55),
    ("Central African Rep.", "CF", "\U0001f1e8\U0001f1eb", "Common", 45, 45, 55, 45),
    ("Chad", "TD", "\U0001f1f9\U0001f1e9", "Common", 50, 45, 55, 45),
    ("Comoros", "KM", "\U0001f1f0\U0001f1f2", "Common", 35, 40, 40, 50),
    ("Congo", "CG", "\U0001f1e8\U0001f1ec", "Common", 50, 45, 55, 50),
    ("DRC", "CD", "\U0001f1e8\U0001f1e9", "Common", 55, 45, 60, 50),
    ("Cyprus", "CY", "\U0001f1e8\U0001f1fe", "Common", 45, 50, 50, 60),
    ("Djibouti", "DJ", "\U0001f1e9\U0001f1ef", "Common", 40, 40, 45, 50),
    ("East Timor", "TL", "\U0001f1f9\U0001f1f1", "Common", 40, 40, 45, 50),
    ("El Salvador", "SV", "\U0001f1f8\U0001f1fb", "Common", 45, 45, 50, 55),
    ("Equatorial Guinea", "GQ", "\U0001f1ec\U0001f1f6", "Common", 40, 45, 50, 45),
    ("Eritrea", "ER", "\U0001f1ea\U0001f1f7", "Common", 45, 45, 50, 45),
    ("Eswatini", "SZ", "\U0001f1f8\U0001f1ff", "Common", 40, 45, 45, 50),
    ("Fiji", "FJ", "\U0001f1eb\U0001f1ef", "Common", 40, 45, 45, 55),
    ("Gabon", "GA", "\U0001f1ec\U0001f1e6", "Common", 45, 50, 50, 45),
    ("Gambia", "GM", "\U0001f1ec\U0001f1f2", "Common", 40, 40, 45, 50),
    ("Georgia", "GE", "\U0001f1ec\U0001f1ea", "Common", 55, 50, 55, 55),
    ("Grenada", "GD", "\U0001f1ec\U0001f1e9", "Common", 35, 40, 40, 50),
    ("Guinea", "GN", "\U0001f1ec\U0001f1f3", "Common", 45, 40, 50, 45),
    ("Guinea-Bissau", "GW", "\U0001f1ec\U0001f1fc", "Common", 35, 40, 40, 45),
    ("Guyana", "GY", "\U0001f1ec\U0001f1fe", "Common", 40, 45, 45, 50),
    ("Haiti", "HT", "\U0001f1ed\U0001f1f9", "Common", 45, 40, 50, 50),
    ("Jamaica", "JM", "\U0001f1ef\U0001f1f2", "Common", 45, 40, 45, 60),
    ("Kiribati", "KI", "\U0001f1f0\U0001f1ee", "Common", 35, 40, 40, 45),
    ("Kyrgyzstan", "KG", "\U0001f1f0\U0001f1ec", "Common", 50, 50, 55, 55),
    ("Laos", "LA", "\U0001f1f1\U0001f1e6", "Common", 45, 45, 50, 55),
    ("Lesotho", "LS", "\U0001f1f1\U0001f1f8", "Common", 40, 45, 45, 45),
    ("Liberia", "LR", "\U0001f1f1\U0001f1f7", "Common", 40, 40, 45, 45),
    ("Liechtenstein", "LI", "\U0001f1f1\U0001f1ee", "Common", 30, 55, 35, 60),
    ("Malawi", "MW", "\U0001f1f2\U0001f1fc", "Common", 40, 40, 50, 45),
    ("Maldives", "MV", "\U0001f1f2\U0001f1fb", "Common", 30, 45, 35, 55),
    ("Mali", "ML", "\U0001f1f2\U0001f1f1", "Common", 45, 45, 50, 45),
    ("Malta", "MT", "\U0001f1f2\U0001f1f9", "Common", 35, 50, 40, 60),
    ("Marshall Islands", "MH", "\U0001f1f2\U0001f1ed", "Common", 30, 35, 35, 45),
    ("Mauritania", "MR", "\U0001f1f2\U0001f1f7", "Common", 40, 45, 50, 40),
    ("Mauritius", "MU", "\U0001f1f2\U0001f1fa", "Common", 35, 45, 40, 55),
    ("Micronesia", "FM", "\U0001f1eb\U0001f1f2", "Common", 30, 35, 35, 45),
    ("Moldova", "MD", "\U0001f1f2\U0001f1e9", "Common", 50, 45, 55, 50),
    ("Monaco", "MC", "\U0001f1f2\U0001f1e8", "Common", 30, 55, 35, 65),
    ("Montenegro", "ME", "\U0001f1f2\U0001f1ea", "Common", 50, 50, 50, 55),
    ("Mozambique", "MZ", "\U0001f1f2\U0001f1ff", "Common", 50, 45, 55, 45),
    ("Namibia", "NA", "\U0001f1f3\U0001f1e6", "Common", 45, 50, 50, 50),
    ("Nauru", "NR", "\U0001f1f3\U0001f1f7", "Common", 30, 35, 35, 40),
    ("Niger", "NE", "\U0001f1f3\U0001f1ea", "Common", 45, 40, 55, 40),
    ("North Macedonia", "MK", "\U0001f1f2\U0001f1f0", "Common", 50, 50, 50, 55),
    ("Palau", "PW", "\U0001f1f5\U0001f1fc", "Common", 30, 35, 35, 45),
    ("Papua New Guinea", "PG", "\U0001f1f5\U0001f1ec", "Common", 45, 40, 50, 45),
    ("Rwanda", "RW", "\U0001f1f7\U0001f1fc", "Common", 45, 40, 55, 50),
    ("Samoa", "WS", "\U0001f1fc\U0001f1f8", "Common", 40, 40, 45, 50),
    ("San Marino", "SM", "\U0001f1f8\U0001f1f2", "Common", 30, 50, 35, 60),
    ("Sao Tome", "ST", "\U0001f1f8\U0001f1f9", "Common", 30, 35, 35, 45),
    ("Senegal", "SN", "\U0001f1f8\U0001f1f3", "Common", 45, 45, 50, 50),
    ("Seychelles", "SC", "\U0001f1f8\U0001f1e8", "Common", 30, 45, 35, 55),
    ("Sierra Leone", "SL", "\U0001f1f8\U0001f1f1", "Common", 40, 40, 45, 45),
    ("Solomon Islands", "SB", "\U0001f1f8\U0001f1e7", "Common", 35, 35, 40, 45),
    ("Somalia", "SO", "\U0001f1f8\U0001f1f4", "Common", 45, 40, 50, 40),
    ("South Sudan", "SS", "\U0001f1f8\U0001f1f8", "Common", 50, 40, 55, 45),
    ("Suriname", "SR", "\U0001f1f8\U0001f1f7", "Common", 40, 45, 45, 45),
    ("Tajikistan", "TJ", "\U0001f1f9\U0001f1ef", "Common", 50, 45, 55, 50),
    ("Togo", "TG", "\U0001f1f9\U0001f1ec", "Common", 40, 40, 50, 45),
    ("Tonga", "TO", "\U0001f1f9\U0001f1f4", "Common", 40, 40, 45, 45),
    ("Trinidad", "TT", "\U0001f1f9\U0001f1f9", "Common", 40, 45, 45, 55),
    ("Turkmenistan", "TM", "\U0001f1f9\U0001f1f2", "Common", 50, 50, 55, 50),
    ("Tuvalu", "TV", "\U0001f1f9\U0001f1fb", "Common", 30, 35, 35, 40),
    ("Vanuatu", "VU", "\U0001f1fb\U0001f1fa", "Common", 35, 35, 40, 45),
    ("Vatican City", "VA", "\U0001f1fb\U0001f1e6", "Common", 20, 60, 30, 55),
    ("Zambia", "ZM", "\U0001f1ff\U0001f1f2", "Common", 50, 45, 55, 50),
]

RARITY_ORDER = {"Common": 0, "Uncommon": 1, "Rare": 2, "Epic": 3, "Legendary": 4}
RARITY_COLORS = {
    "Common": 0x9d9d9d, "Uncommon": 0x1eff00, "Rare": 0x0070dd,
    "Epic": 0xa335ee, "Legendary": 0xff8000,
}
RARITY_WEIGHTS = {"Common": 40, "Uncommon": 30, "Rare": 18, "Epic": 9, "Legendary": 3}

# ─── Helpers ────────────────────────────────────────────────────────

def _roll_iv():
    return random.randint(0, 31)

def _total_iv(b: dict) -> int:
    return b.get("attack_iv", 0) + b.get("defense_iv", 0) + b.get("hp_iv", 0) + b.get("speed_iv", 0)

def _iv_percent(b: dict) -> int:
    return int(_total_iv(b) / 124 * 100)

def _xp_for_level(level: int) -> int:
    return level * level * 100

# ─── DB ─────────────────────────────────────────────────────────────

_get_db_func = None

def _init_db(get_db_func):
    global _get_db_func
    _get_db_func = get_db_func

def _db():
    return _get_db_func()

# ─── Paginator ──────────────────────────────────────────────────────

class SimplePaginator(discord.ui.View):
    def __init__(self, pages, user_id):
        super().__init__(timeout=60)
        self.pages = pages
        self.user_id = user_id
        self.current = 0

    @discord.ui.button(label="\u25c0", style=discord.ButtonStyle.grey)
    async def prev(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message(":x: Not your menu!", ephemeral=True)
        self.current = (self.current - 1) % len(self.pages)
        await interaction.response.edit_message(embed=self.pages[self.current])

    @discord.ui.button(label="\u25b6", style=discord.ButtonStyle.grey)
    async def next(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user_id:
            return await interaction.response.send_message(":x: Not your menu!", ephemeral=True)
        self.current = (self.current + 1) % len(self.pages)
        await interaction.response.edit_message(embed=self.pages[self.current])

def _ensure_countryballs():
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM ballsdex_countryballs")
    if cur.fetchone()[0] > 0:
        conn.close()
        return
    for name, code, emoji, rarity, atk, df, hp, spd in COUNTRY_DATA:
        cur.execute(
            "INSERT INTO ballsdex_countryballs (name, country_code, emoji, rarity, attack_base, defense_base, hp_base, speed_base) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (name, code, emoji, rarity, atk, df, hp, spd),
        )
    conn.commit()
    conn.close()
    print(f"[BALLSDEX] Seeded {len(COUNTRY_DATA)} countryballs")

# ─── Spawn System ───────────────────────────────────────────────────

SPAWNED_MESSAGES: dict[int, dict] = {}

async def spawn_ball(guild: discord.Guild, channel: discord.TextChannel):
    ensure_countryballs_called = getattr(spawn_ball, "_seeded", False)
    if not ensure_countryballs_called:
        _ensure_countryballs()
        spawn_ball._seeded = True

    conn = _db()
    cur = conn.cursor()

    # Pick random countryball weighted by rarity
    cur.execute("SELECT id, name, country_code, emoji, rarity, attack_base, defense_base, hp_base, speed_base FROM ballsdex_countryballs")
    all_balls = cur.fetchall()
    if not all_balls:
        conn.close()
        return
    weights = [RARITY_WEIGHTS.get(b[4], 10) for b in all_balls]
    chosen = random.choices(all_balls, weights=weights, k=1)[0]
    b_id, b_name, b_code, b_emoji, b_rarity, b_atk, b_def, b_hp, b_spd = chosen

    shiny = 1 if random.random() < 0.02 else 0
    atk_iv = _roll_iv()
    def_iv = _roll_iv()
    hp_iv = _roll_iv()
    spd_iv = _roll_iv()
    conn.close()

    embed = discord.Embed(
        title=f"A wild {b_name} appeared! {b_emoji}",
        description=f"**Rarity:** {b_rarity}{' ✨ SHINY' if shiny else ''}\n"
                    f"**ATK:** {b_atk + atk_iv} | **DEF:** {b_def + def_iv} | "
                    f"**HP:** {b_hp + hp_iv} | **SPD:** {b_spd + spd_iv}",
        color=RARITY_COLORS.get(b_rarity, 0x9d9d9d),
    )
    embed.set_footer(text="Click Catch to add this countryball to your collection!")

    view = CatchView(b_id, shiny, atk_iv, def_iv, hp_iv, spd_iv)
    msg = await channel.send(embed=embed, view=view)
    view.message = msg

    # Track spawned ball in DB
    conn2 = _db()
    conn2.execute("INSERT OR REPLACE INTO ballsdex_spawned (message_id, guild_id, channel_id, countryball_id, shiny, caught, spawned_at) VALUES (?, ?, ?, ?, ?, 0, ?)",
                  (msg.id, guild.id, channel.id, b_id, shiny, time.time()))
    conn2.commit()
    conn2.close()

    key = f"{guild.id}:{channel.id}"
    if key not in SPAWNED_MESSAGES:
        SPAWNED_MESSAGES[key] = {}
    SPAWNED_MESSAGES[key][msg.id] = {
        "countryball_id": b_id,
        "shiny": shiny,
        "atk_iv": atk_iv, "def_iv": def_iv, "hp_iv": hp_iv, "spd_iv": spd_iv,
        "caught": False,
    }

    # Auto-disable after 5 minutes
    await asyncio.sleep(300)
    try:
        await msg.edit(view=None)
    except Exception:
        pass
    SPAWNED_MESSAGES.get(key, {}).pop(msg.id, None)


class CatchView(discord.ui.View):
    def __init__(self, countryball_id, shiny, atk_iv, def_iv, hp_iv, spd_iv):
        super().__init__(timeout=300)
        self.countryball_id = countryball_id
        self.shiny = shiny
        self.atk_iv = atk_iv
        self.def_iv = def_iv
        self.hp_iv = hp_iv
        self.spd_iv = spd_iv
        self.message = None

    @discord.ui.button(label="\U0001f3af Catch!", style=discord.ButtonStyle.green)
    async def catch_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        key = f"{interaction.guild_id}:{interaction.channel_id}"
        spawned = SPAWNED_MESSAGES.get(key, {}).get(self.message.id) if self.message else None
        if not spawned or spawned.get("caught"):
            await interaction.response.send_message(":x: This ball was already caught!", ephemeral=True)
            return
        spawned["caught"] = True

        conn = _db()
        conn.execute(
            "INSERT INTO ballsdex_instances (user_id, guild_id, countryball_id, attack_iv, defense_iv, hp_iv, speed_iv, shiny, caught_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (interaction.user.id, interaction.guild_id, self.countryball_id, self.atk_iv, self.def_iv, self.hp_iv, self.spd_iv, self.shiny, time.time()),
        )
        conn.commit()
        conn.close()

        # Update spawned record
        conn2 = _db()
        conn2.execute("UPDATE ballsdex_spawned SET caught = 1 WHERE message_id = ?", (self.message.id,))
        conn2.commit()
        conn2.close()

        embed = self.message.embeds[0] if self.message else None
        if embed:
            embed.title = f"Caught! {embed.title.replace('A wild ', '').replace(' appeared!', '')}"
        await self.message.edit(view=None, embed=embed)
        await interaction.response.send_message(
            f"\U0001f389 You caught a countryball! {'✨ SHINY! ' if self.shiny else ''}"
            f"Check your collection with `/ballsdex collection`!",
            ephemeral=True,
        )

        self.stop()

    async def on_timeout(self):
        if self.message:
            try:
                await self.message.edit(view=None)
            except Exception:
                pass


# ─── Admin / Spawn Loop ─────────────────────────────────────────────

async def ballsdex_spawn_loop(bot):
    await bot.wait_until_ready()
    while not bot.is_closed():
        try:
            conn = _db()
            cur = conn.cursor()
            cur.execute("SELECT guild_id, channel_id FROM ballsdex_spawn_settings WHERE spawn_enabled = 1")
            rows = cur.fetchall()
            conn.close()
            for guild_id, channel_id in rows:
                guild = bot.get_guild(guild_id)
                if not guild:
                    continue
                channel = guild.get_channel(channel_id)
                if not channel:
                    continue
                await spawn_ball(guild, channel)
                await asyncio.sleep(30)
        except Exception as e:
            print(f"[BALLSDEX] spawn loop error: {e}")
            await asyncio.sleep(60)

# ─── Image Generation ───────────────────────────────────────────────

async def generate_ball_card(countryball: dict, instance: dict = None) -> Optional[io.BytesIO]:
    try:
        import aiohttp
        w, h = 400, 300
        img = Image.new("RGBA", (w, h), (30, 30, 50, 255))
        draw = ImageDraw.Draw(img)

        font_title = None
        font_stat = None
        font_small = None
        try:
            font_title = ImageFont.truetype("arial.ttf", 28)
            font_stat = ImageFont.truetype("arial.ttf", 18)
            font_small = ImageFont.truetype("arial.ttf", 14)
        except Exception:
            font_title = ImageFont.load_default()
            font_stat = font_title
            font_small = font_title

        # Download flag
        flag_url = f"https://flagcdn.com/64x48/{countryball['country_code'].lower()}.png"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(flag_url, timeout=10) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        flag = Image.open(io.BytesIO(data)).convert("RGBA")
                        flag = flag.resize((120, 90), Image.LANCZOS)
                        img.paste(flag, (20, 40), flag)
        except Exception:
            pass

        # Name + emoji
        name_text = f"{countryball['emoji']} {countryball['name']}"
        if font_title:
            draw.text((160, 40), name_text, fill=(255, 255, 255), font=font_title)

        # Rarity
        rarity = countryball.get("rarity", "Common")
        rarity_color = RARITY_COLORS.get(rarity, 0x9d9d9d)
        if font_stat:
            draw.text((160, 80), f"Rarity: {rarity}", fill=rarity_color, font=font_stat)

        shiny = instance and instance.get("shiny")
        if shiny:
            draw.text((160, 105), "✨ SHINY", fill=(255, 215, 0), font=font_stat)

        # Stats
        if instance:
            y = 160
            stats = [
                ("ATK", countryball.get("attack_base", 50) + instance.get("attack_iv", 0), instance.get("attack_iv", 0)),
                ("DEF", countryball.get("defense_base", 50) + instance.get("defense_iv", 0), instance.get("defense_iv", 0)),
                ("HP", countryball.get("hp_base", 50) + instance.get("hp_iv", 0), instance.get("hp_iv", 0)),
                ("SPD", countryball.get("speed_base", 50) + instance.get("speed_iv", 0), instance.get("speed_iv", 0)),
            ]
            for label, total, iv in stats:
                if font_small:
                    draw.text((20, y), f"{label}: {total} (+{iv})", fill=(200, 200, 220), font=font_small)
                bar_w = 200
                bar_h = 10
                fill_w = int(bar_w * min(total / 200, 1))
                draw.rectangle((140, y, 140 + bar_w, y + bar_h), fill=(50, 50, 70))
                draw.rectangle((140, y, 140 + fill_w, y + bar_h), fill=rarity_color)
                y += 25

            iv_pct = int((instance.get("attack_iv", 0) + instance.get("defense_iv", 0) + instance.get("hp_iv", 0) + instance.get("speed_iv", 0)) / 124 * 100)
            if font_small:
                draw.text((20, y + 5), f"IV Total: {iv_pct}%", fill=(255, 255, 255), font=font_small)

        buf = io.BytesIO()
        img.save(buf, "PNG")
        buf.seek(0)
        return buf
    except Exception as e:
        print(f"[BALLSDEX] card gen error: {e}")
        return None


# ─── Commands ───────────────────────────────────────────────────────

ballsdex_group = app_commands.Group(name="ballsdex", description="Countryball collection game", guild_only=True)


@ballsdex_group.command(name="collection", description="View your countryball collection")
@app_commands.describe(user="User to view collection of (optional)")
async def cmd_collection(interaction: discord.Interaction, user: discord.User = None):
    target = user or interaction.user
    await interaction.response.defer()
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "SELECT ci.id, ci.countryball_id, ci.attack_iv, ci.defense_iv, ci.hp_iv, ci.speed_iv, ci.shiny, ci.favorite, cb.name, cb.emoji, cb.rarity "
        "FROM ballsdex_instances ci JOIN ballsdex_countryballs cb ON ci.countryball_id = cb.id "
        "WHERE ci.user_id = ? AND ci.guild_id = ? ORDER BY ci.favorite DESC, cb.rarity DESC, ci.id",
        (target.id, interaction.guild_id),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        await interaction.followup.send(f"{target.mention} has no countryballs yet! Wait for one to spawn!")
        return

    total_balls = len(rows)
    favs = sum(1 for r in rows if r[7])
    shiny_count = sum(1 for r in rows if r[6])

    items = []
    for r in rows:
        r_dict = {"id": r[0], "countryball_id": r[1], "attack_iv": r[2], "defense_iv": r[3], "hp_iv": r[4], "speed_iv": r[5], "shiny": r[6], "favorite": r[7]}
        stars = "\u2b50" if r[7] else ""
        shiny_tag = "✨" if r[6] else ""
        iv = _iv_percent(r_dict)
        items.append(f"{stars}{shiny_tag} **{r[9]} {r[8]}** — {r[10]} — IV: {iv}%")

    pages = []
    for i in range(0, len(items), 10):
        chunk = items[i:i+10]
        embed = discord.Embed(
            title=f"{target.display_name}'s Collection",
            description="\n".join(chunk),
            color=0x3498db,
        )
        embed.set_footer(text=f"Total: {total_balls} | Shiny: {shiny_count} | Favorites: {favs}")
        pages.append(embed)

    if len(pages) == 1:
        await interaction.followup.send(embed=pages[0])
    else:
        view = SimplePaginator(pages, interaction.user.id)
        await interaction.followup.send(embed=pages[0], view=view)


@ballsdex_group.command(name="stats", description="View stats of a specific countryball")
@app_commands.describe(ball_id="The ID of the ball from your collection")
async def cmd_stats(interaction: discord.Interaction, ball_id: int):
    await interaction.response.defer()
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "SELECT ci.id, ci.countryball_id, ci.attack_iv, ci.defense_iv, ci.hp_iv, ci.speed_iv, ci.shiny, ci.favorite, ci.level, ci.xp, "
        "cb.name, cb.emoji, cb.rarity, cb.attack_base, cb.defense_base, cb.hp_base, cb.speed_base "
        "FROM ballsdex_instances ci JOIN ballsdex_countryballs cb ON ci.countryball_id = cb.id WHERE ci.id = ? AND ci.user_id = ?",
        (ball_id, interaction.user.id),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        await interaction.followup.send(":x: Ball not found or not yours!")
        return

    instance = {"id": row[0], "attack_iv": row[2], "defense_iv": row[3], "hp_iv": row[4], "speed_iv": row[5], "shiny": row[6], "favorite": row[7], "level": row[8], "xp": row[9]}
    cb = {"name": row[10], "emoji": row[11], "rarity": row[12], "attack_base": row[13], "defense_base": row[14], "hp_base": row[15], "speed_base": row[16], "country_code": next((c[1] for c in COUNTRY_DATA if c[0] == row[10]), "UN").lower()}
    iv_pct = _iv_percent(instance)

    card = await generate_ball_card(cb, instance)
    embed = discord.Embed(
        title=f"{cb['emoji']} {cb['name']}",
        description=f"**Rarity:** {cb['rarity']}{' ✨ SHINY' if instance['shiny'] else ''}\n"
                    f"**Level:** {instance['level']} | **IV:** {iv_pct}%\n"
                    f"{'\u2b50 Favorite' if instance['favorite'] else ''}",
        color=RARITY_COLORS.get(cb['rarity'], 0x9d9d9d),
    )
    embed.add_field(name="Stats", value=
        f"ATK: {cb['attack_base'] + instance['attack_iv']} (+{instance['attack_iv']} IV)\n"
        f"DEF: {cb['defense_base'] + instance['defense_iv']} (+{instance['defense_iv']} IV)\n"
        f"HP: {cb['hp_base'] + instance['hp_iv']} (+{instance['hp_iv']} IV)\n"
        f"SPD: {cb['speed_base'] + instance['speed_iv']} (+{instance['speed_iv']} IV)"
    )
    embed.set_footer(text=f"ID: {instance['id']} | Ball #{instance['id']}")

    file = discord.File(card, "ball.png") if card else None
    if file:
        embed.set_image(url="attachment://ball.png")
    await interaction.followup.send(embed=embed, file=file)


@ballsdex_group.command(name="favorite", description="Toggle favorite on a ball")
@app_commands.describe(ball_id="The ID of the ball to favorite/unfavorite")
async def cmd_favorite(interaction: discord.Interaction, ball_id: int):
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT favorite FROM ballsdex_instances WHERE id = ? AND user_id = ?", (ball_id, interaction.user.id))
    row = cur.fetchone()
    if not row:
        await interaction.response.send_message(":x: Ball not found or not yours!", ephemeral=True)
        conn.close()
        return
    new_fav = 0 if row[0] else 1
    cur.execute("UPDATE ballsdex_instances SET favorite = ? WHERE id = ?", (new_fav, ball_id))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f"{'⭐' if new_fav else '⭕'} Ball **{ball_id}** {'favorited' if new_fav else 'unfavorited'}!")


@ballsdex_group.command(name="leaderboard", description="Top collectors in this server")
async def cmd_leaderboard(interaction: discord.Interaction):
    await interaction.response.defer()
    conn = _db()
    cur = conn.cursor()
    cur.execute(
        "SELECT user_id, COUNT(*) as count FROM ballsdex_instances WHERE guild_id = ? GROUP BY user_id ORDER BY count DESC LIMIT 15",
        (interaction.guild_id,),
    )
    rows = cur.fetchall()
    conn.close()
    if not rows:
        await interaction.followup.send("No one has caught any balls yet!")
        return

    lines = []
    medals = ["\U0001f947", "\U0001f948", "\U0001f949"]
    for i, (uid, count) in enumerate(rows):
        member = interaction.guild.get_member(uid)
        name = member.display_name if member else f"<@{uid}>"
        medal = medals[i] if i < 3 else f"{i+1}."
        lines.append(f"{medal} **{name}** — {count} balls")

    embed = discord.Embed(title="\U0001f3c6 Countryball Leaderboard", description="\n".join(lines), color=0xf1c40f)
    await interaction.followup.send(embed=embed)


@ballsdex_group.command(name="trade", description="Trade a ball with another user")
@app_commands.describe(user="User to trade with", ball_id="Your ball's ID")
async def cmd_trade(interaction: discord.Interaction, user: discord.User, ball_id: int):
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT id FROM ballsdex_instances WHERE id = ? AND user_id = ?", (ball_id, interaction.user.id))
    if not cur.fetchone():
        await interaction.response.send_message(":x: Ball not found or not yours!", ephemeral=True)
        conn.close()
        return

    trade_id = int(time.time())
    cur.execute("INSERT INTO ballsdex_trades (id, user1_id, user2_id, status, created_at) VALUES (?, ?, ?, 'pending', ?)",
                (trade_id, interaction.user.id, user.id, time.time()))
    cur.execute("INSERT INTO ballsdex_trade_items (trade_id, user_id, instance_id) VALUES (?, ?, ?)",
                (trade_id, interaction.user.id, ball_id))
    conn.commit()
    conn.close()

    view = TradeAcceptView(trade_id, interaction.user.id, user.id)
    await interaction.response.send_message(
        f"\U0001f91d {interaction.user.mention} wants to trade with {user.mention}!\n"
        f"Ball ID: **{ball_id}**\n"
        f"{user.mention}, click Accept to proceed, then offer your ball with `/ballsdex trade-offer {trade_id} <ball_id>`",
        view=view,
    )


class TradeAcceptView(discord.ui.View):
    def __init__(self, trade_id, user1_id, user2_id):
        super().__init__(timeout=120)
        self.trade_id = trade_id
        self.user1_id = user1_id
        self.user2_id = user2_id

    @discord.ui.button(label="\u2705 Accept Trade", style=discord.ButtonStyle.green)
    async def accept(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id != self.user2_id:
            await interaction.response.send_message(":x: Only the trade recipient can accept!", ephemeral=True)
            return
        conn = _db()
        conn.execute("UPDATE ballsdex_trades SET status = 'accepted' WHERE id = ?", (self.trade_id,))
        conn.commit()
        conn.close()
        await interaction.response.send_message(f"\u2705 Trade accepted! {interaction.user.mention}, use `/ballsdex trade-offer {self.trade_id} <your_ball_id>` to offer your ball.")
        self.stop()

    @discord.ui.button(label="\u274c Decline", style=discord.ButtonStyle.red)
    async def decline(self, interaction: discord.Interaction, button: discord.ui.Button):
        if interaction.user.id not in (self.user1_id, self.user2_id):
            await interaction.response.send_message(":x: Not your trade!", ephemeral=True)
            return
        conn = _db()
        conn.execute("UPDATE ballsdex_trades SET status = 'declined' WHERE id = ?", (self.trade_id,))
        conn.commit()
        conn.close()
        await interaction.response.send_message("\u274c Trade declined.")
        self.stop()


@ballsdex_group.command(name="trade-offer", description="Offer your ball in an accepted trade")
@app_commands.describe(trade_id="The trade ID", ball_id="Your ball's ID to trade")
async def cmd_trade_offer(interaction: discord.Interaction, trade_id: int, ball_id: int):
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT status, user1_id, user2_id FROM ballsdex_trades WHERE id = ?", (trade_id,))
    trade = cur.fetchone()
    if not trade or trade[0] != "accepted":
        await interaction.response.send_message(":x: Trade not found or not accepted yet!", ephemeral=True)
        conn.close()
        return
    if interaction.user.id not in (trade[1], trade[2]):
        await interaction.response.send_message(":x: Not your trade!", ephemeral=True)
        conn.close()
        return

    other_id = trade[2] if interaction.user.id == trade[1] else trade[1]
    cur.execute("SELECT id FROM ballsdex_instances WHERE id = ? AND user_id = ?", (ball_id, interaction.user.id))
    if not cur.fetchone():
        await interaction.response.send_message(":x: Ball not found or not yours!", ephemeral=True)
        conn.close()
        return

    # Check if they already offered
    cur.execute("SELECT id FROM ballsdex_trade_items WHERE trade_id = ? AND user_id = ?", (trade_id, interaction.user.id))
    if cur.fetchone():
        cur.execute("UPDATE ballsdex_trade_items SET instance_id = ? WHERE trade_id = ? AND user_id = ?",
                    (ball_id, trade_id, interaction.user.id))
    else:
        cur.execute("INSERT INTO ballsdex_trade_items (trade_id, user_id, instance_id) VALUES (?, ?, ?)",
                    (trade_id, interaction.user.id, ball_id))
    conn.commit()

    # Check if both offered
    cur.execute("SELECT COUNT(*) FROM ballsdex_trade_items WHERE trade_id = ?", (trade_id,))
    if cur.fetchone()[0] == 2:
        cur.execute("SELECT user_id, instance_id FROM ballsdex_trade_items WHERE trade_id = ?", (trade_id,))
        items = cur.fetchall()
        i1 = next(i for i in items if i[0] != interaction.user.id)
        i2 = next(i for i in items if i[0] == interaction.user.id)
        # Swap ownership
        cur.execute("UPDATE ballsdex_instances SET user_id = ? WHERE id = ?", (i1[0], i2[1]))
        cur.execute("UPDATE ballsdex_instances SET user_id = ? WHERE id = ?", (i2[0], i1[1]))
        cur.execute("UPDATE ballsdex_trades SET status = 'completed' WHERE id = ?", (trade_id,))
        conn.commit()
        conn.close()
        await interaction.response.send_message(f"\U0001f91d Trade completed! Balls swapped!")
    else:
        conn.close()
        await interaction.response.send_message(f"\u2705 You offered ball **{ball_id}**. Waiting for the other user to offer theirs.")


@ballsdex_group.command(name="sell", description="Sell a duplicate ball for coins")
@app_commands.describe(ball_id="The ID of the ball to sell")
async def cmd_sell(interaction: discord.Interaction, ball_id: int):
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT id, countryball_id, shiny FROM ballsdex_instances WHERE id = ? AND user_id = ?", (ball_id, interaction.user.id))
    ball = cur.fetchone()
    if not ball:
        await interaction.response.send_message(":x: Ball not found or not yours!", ephemeral=True)
        conn.close()
        return

    rarity_map = {"Common": 10, "Uncommon": 25, "Rare": 50, "Epic": 100, "Legendary": 250}
    cur.execute("SELECT rarity FROM ballsdex_countryballs WHERE id = ?", (ball[1],))
    rarity = cur.fetchone()[0]
    base_price = rarity_map.get(rarity, 10)
    if ball[2]:
        base_price *= 3
    cur.execute("DELETE FROM ballsdex_instances WHERE id = ?", (ball_id,))
    cur.execute("INSERT INTO ballsdex_economy (user_id, guild_id, coins) VALUES (?, ?, ?) ON CONFLICT(user_id, guild_id) DO UPDATE SET coins = coins + ?",
                (interaction.user.id, interaction.guild_id, base_price, base_price))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f"\U0001f4b0 Sold ball **{ball_id}** for **{base_price} coins**!")


@ballsdex_group.command(name="balance", description="Check your coin balance")
async def cmd_balance(interaction: discord.Interaction):
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT coins FROM ballsdex_economy WHERE user_id = ? AND guild_id = ?", (interaction.user.id, interaction.guild_id))
    row = cur.fetchone()
    conn.close()
    coins = row[0] if row else 0
    await interaction.response.send_message(f"\U0001f4b0 **{interaction.user.display_name}** has **{coins} coins**.")


@ballsdex_group.command(name="pokedex", description="View all available countryballs and which you've caught")
async def cmd_pokedex(interaction: discord.Interaction):
    await interaction.response.defer()
    conn = _db()
    cur = conn.cursor()
    cur.execute("SELECT id, name, emoji, rarity FROM ballsdex_countryballs ORDER BY rarity, name")
    all_balls = cur.fetchall()
    cur.execute("SELECT DISTINCT countryball_id FROM ballsdex_instances WHERE user_id = ? AND guild_id = ?",
                (interaction.user.id, interaction.guild_id))
    caught_ids = {r[0] for r in cur.fetchall()}
    conn.close()

    total = len(all_balls)
    caught_count = len(caught_ids)

    by_rarity = {}
    for b in all_balls:
        by_rarity.setdefault(b[3], []).append(b)

    lines = []
    for rarity in ["Common", "Uncommon", "Rare", "Epic", "Legendary"]:
        balls = by_rarity.get(rarity, [])
        if not balls:
            continue
        lines.append(f"\n**{rarity}** ({len(balls)})")
        for b in balls:
            caught = "\u2705" if b[0] in caught_ids else "\u274c"
            lines.append(f"{caught} {b[2]} {b[1]}")

    embed = discord.Embed(
        title=f"\U0001f4ca Pokédex — {caught_count}/{total}",
        description="".join(lines[:50]),
        color=0x2ecc71,
    )
    embed.set_footer(text=f"You've caught {caught_count} out of {total} countryballs!")
    await interaction.followup.send(embed=embed)


# ─── Admin Commands ─────────────────────────────────────────────────

admin_group = app_commands.Group(name="bsetup", description="Ballsdex admin setup", guild_only=True)


@admin_group.command(name="spawn-channel", description="Set a channel for balls to spawn in")
@app_commands.describe(channel="The channel for spawning")
async def cmd_spawn_channel(interaction: discord.Interaction, channel: discord.TextChannel):
    conn = _db()
    conn.execute("INSERT INTO ballsdex_spawn_settings (guild_id, channel_id) VALUES (?, ?, 1, 30) ON CONFLICT(guild_id, channel_id) DO UPDATE SET spawn_enabled = 1",
                 (interaction.guild_id, channel.id))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f"\U0001f4ac Balls will now spawn in {channel.mention} every ~30s!")


@admin_group.command(name="remove-channel", description="Remove a spawn channel")
@app_commands.describe(channel="The channel to stop spawning in")
async def cmd_remove_channel(interaction: discord.Interaction, channel: discord.TextChannel):
    conn = _db()
    conn.execute("DELETE FROM ballsdex_spawn_settings WHERE guild_id = ? AND channel_id = ?",
                 (interaction.guild_id, channel.id))
    conn.commit()
    conn.close()
    await interaction.response.send_message(f"\U0001f4ac Stopped spawning in {channel.mention}.")


@admin_group.command(name="spawn-now", description="Force-spawn a random ball in a channel")
@app_commands.describe(channel="Channel to spawn in")
async def cmd_spawn_now(interaction: discord.Interaction, channel: discord.TextChannel):
    await interaction.response.defer()
    await spawn_ball(interaction.guild, channel)
    await interaction.followup.send(f"\U0001f4ac Spawned a ball in {channel.mention}!")
