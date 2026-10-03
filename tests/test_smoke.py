from pathlib import Path

import yaml

import allocation_lab

ROOT = Path(__file__).resolve().parents[1]

SLEEVES = {"idx_equities", "us_equities", "gold", "crypto"}
UNIVERSE_KEYS = {"base_currency", "sleeves", "fx", "benchmarks"}
COSTS_KEYS = {"confirmed", "as_of", "slippage_pct", "cash"} | SLEEVES
LOCAL_KEYS = {"monthly_contribution_idr", "current_holdings"}


def load_config(name):
    with open(ROOT / "config" / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_package_imports():
    assert allocation_lab.__version__


def test_universe_config_structure():
    cfg = load_config("universe.yaml")
    assert UNIVERSE_KEYS <= cfg.keys()
    assert cfg["base_currency"] == "IDR"
    assert set(cfg["sleeves"]) == SLEEVES
    for sleeve in cfg["sleeves"].values():
        assert {"currency", "members"} <= sleeve.keys()


def test_costs_config_structure():
    cfg = load_config("costs.yaml")
    assert COSTS_KEYS <= cfg.keys()


def test_local_example_structure():
    cfg = load_config("local.example.yaml")
    assert LOCAL_KEYS <= cfg.keys()
