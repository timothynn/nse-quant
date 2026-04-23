"""Central configuration for NSE Quant."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytz

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
SAMPLE_DATA_DIR = PACKAGE_ROOT / "data" / "sample"
CACHE_DIR = Path(os.getenv("NSE_QUANT_CACHE_DIR", REPO_ROOT / "data" / "cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

EAT = pytz.timezone("Africa/Nairobi")

# Top NSE constituents with their yfinance-compatible tickers (.NR suffix for Nairobi).
# Sector classification follows NSE industry groupings.
NSE_UNIVERSE: dict[str, dict[str, str]] = {
    "SCOM.NR":  {"name": "Safaricom PLC",              "sector": "Telecom"},
    "EQTY.NR":  {"name": "Equity Group Holdings",      "sector": "Banking"},
    "KCB.NR":   {"name": "KCB Group",                  "sector": "Banking"},
    "COOP.NR":  {"name": "Co-operative Bank",          "sector": "Banking"},
    "ABSA.NR":  {"name": "Absa Bank Kenya",            "sector": "Banking"},
    "SCBK.NR":  {"name": "Standard Chartered Kenya",   "sector": "Banking"},
    "SBIC.NR":  {"name": "Stanbic Holdings",           "sector": "Banking"},
    "NCBA.NR":  {"name": "NCBA Group",                 "sector": "Banking"},
    "DTK.NR":   {"name": "Diamond Trust Bank",         "sector": "Banking"},
    "IMH.NR":   {"name": "I&M Holdings",               "sector": "Banking"},
    "EABL.NR":  {"name": "East African Breweries",     "sector": "Consumer"},
    "BAT.NR":   {"name": "British American Tobacco KE","sector": "Consumer"},
    "BAMB.NR":  {"name": "Bamburi Cement",             "sector": "Construction"},
    "KEGN.NR":  {"name": "KenGen",                     "sector": "Energy"},
    "KPLC.NR":  {"name": "Kenya Power & Lighting",     "sector": "Energy"},
    "TOTL.NR":  {"name": "TotalEnergies Marketing KE", "sector": "Energy"},
    "JUB.NR":   {"name": "Jubilee Holdings",           "sector": "Insurance"},
    "BRIT.NR":  {"name": "Britam Holdings",            "sector": "Insurance"},
    "CIC.NR":   {"name": "CIC Insurance Group",        "sector": "Insurance"},
    "UNGA.NR":  {"name": "Unga Group",                 "sector": "Agriculture"},
    "SASN.NR":  {"name": "Sasini PLC",                 "sector": "Agriculture"},
    "KUKZ.NR":  {"name": "Kakuzi",                     "sector": "Agriculture"},
    "NMG.NR":   {"name": "Nation Media Group",         "sector": "Media"},
    "SCAN.NR":  {"name": "Scangroup",                  "sector": "Media"},
    "CTUM.NR":  {"name": "Centum Investment",          "sector": "Investment"},
    "UMME.NR":  {"name": "Umeme Ltd",                  "sector": "Energy"},
    "CARB.NR":  {"name": "Carbacid Investments",       "sector": "Industrial"},
    "CRWN.NR":  {"name": "Crown Paints Kenya",         "sector": "Industrial"},
    "KNRE.NR":  {"name": "Kenya Re",                   "sector": "Insurance"},
    "LIBT.NR":  {"name": "Liberty Kenya Holdings",     "sector": "Insurance"},
}

NSE_INDICES = ["^NSEASI", "^NSE20"]  # NSE All Share, NSE 20 — best-effort via yfinance

# NSE trading holidays (Kenyan public holidays that close the exchange).
# Conservative static list — we ship what's known and documented.
NSE_HOLIDAYS_STATIC: list[str] = [
    "2024-01-01", "2024-03-29", "2024-04-01", "2024-05-01", "2024-06-01",
    "2024-06-17", "2024-10-10", "2024-10-21", "2024-12-12", "2024-12-25", "2024-12-26",
    "2025-01-01", "2025-04-18", "2025-04-21", "2025-05-01", "2025-06-01",
    "2025-06-07", "2025-10-10", "2025-10-20", "2025-12-12", "2025-12-25", "2025-12-26",
    "2026-01-01", "2026-04-03", "2026-04-06", "2026-05-01", "2026-06-01",
    "2026-10-12", "2026-10-20", "2026-12-12", "2026-12-25", "2026-12-26",
]


@dataclass
class Settings:
    universe: dict[str, dict[str, str]] = field(default_factory=lambda: dict(NSE_UNIVERSE))
    indices: list[str] = field(default_factory=lambda: list(NSE_INDICES))
    timezone: str = "Africa/Nairobi"
    base_currency: str = "KES"
    quote_currency: str = "USD"
    # Risk limits
    max_position_weight: float = 0.15
    max_sector_weight: float = 0.35
    max_drawdown_stop: float = 0.25
    # Backtest defaults
    initial_capital: float = 1_000_000.0  # KES
    commission_bps: float = 20.0  # 0.20% round-trip is realistic for NSE retail
    slippage_bps: float = 10.0


settings = Settings()
