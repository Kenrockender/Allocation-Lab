"""
costs.py — Cost model loaded from config/costs.yaml.

CostModel.trade_cost(sleeve, side, gross_value_idr) -> IDR cost of one trade.
CostModel.cash_daily_return() -> daily cash yield multiplier (e.g. 1 + 0.045/365).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "config"


def _load_costs() -> dict:
    with open(CONFIG_DIR / "costs.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class CostModel:
    idx_buy_pct: float
    idx_sell_pct: float
    us_trade_pct: float
    us_fx_spread_pct: float
    gold_spread_pct: float
    crypto_trade_pct: float
    crypto_tax_pct: float
    slippage_pct: float
    cash_annual_yield_pct: float

    # ------------------------------------------------------------------ #
    # Factory                                                              #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_yaml(cls) -> "CostModel":
        cfg = _load_costs()
        idx = cfg["idx_equities"]
        us = cfg["us_equities"]
        gold = cfg["gold"]
        crypto = cfg["crypto"]
        return cls(
            idx_buy_pct=idx["buy_fee_pct"],
            idx_sell_pct=idx["sell_fee_pct"],
            us_trade_pct=us["trading_fee_pct"],
            us_fx_spread_pct=us["fx_spread_pct"],
            gold_spread_pct=gold["spread_pct"],
            crypto_trade_pct=crypto["trading_fee_pct"],
            crypto_tax_pct=crypto["tax_pct"],
            slippage_pct=cfg["slippage_pct"],
            cash_annual_yield_pct=cfg["cash"]["annual_yield_pct"],
        )

    # ------------------------------------------------------------------ #
    # Cost calculation                                                     #
    # ------------------------------------------------------------------ #

    def trade_cost(self, sleeve: str, side: str, gross_value_idr: float) -> float:
        """Return the IDR cost (positive number) of one trade.

        Parameters
        ----------
        sleeve : one of idx_equities | us_equities | gold | crypto
        side   : 'buy' or 'sell'
        gross_value_idr : trade value before fees
        """
        v = abs(gross_value_idr)
        slip = self.slippage_pct / 100

        if sleeve == "idx_equities":
            fee_pct = (self.idx_buy_pct if side == "buy" else self.idx_sell_pct) / 100
            return v * (fee_pct + slip)

        if sleeve == "us_equities":
            fee_pct = self.us_trade_pct / 100
            fx_pct = self.us_fx_spread_pct / 100
            return v * (fee_pct + fx_pct + slip)

        if sleeve == "gold":
            # spread is round-trip, so half on each leg
            half_spread = self.gold_spread_pct / 100 / 2
            return v * (half_spread + slip)

        if sleeve == "crypto":
            trade_pct = self.crypto_trade_pct / 100
            # tax only on sells
            tax_pct = (self.crypto_tax_pct / 100) if side == "sell" else 0.0
            return v * (trade_pct + tax_pct + slip)

        raise ValueError(f"Unknown sleeve: {sleeve!r}")

    def cash_daily_multiplier(self) -> float:
        """Daily growth factor of cash in the money-market fund."""
        return 1.0 + self.cash_annual_yield_pct / 100 / 365
