from pathlib import Path

import pytest

from driftwatch.config import Settings, load_settings

CONFIG = Path(__file__).resolve().parents[1] / "config" / "default.yaml"


@pytest.fixture
def settings() -> Settings:
    return load_settings(CONFIG)
