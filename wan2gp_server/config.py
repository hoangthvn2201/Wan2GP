"""Server configuration, sourced from environment variables.

All knobs have sensible defaults so `python -m wan2gp_server` works
out of the box when the package lives inside the Wan2GP repository.

Environment variables:

    WAN2GP_ROOT                 Path to the Wan2GP installation (default: parent of this package)
    WAN2GP_CLI_ARGS             WanGP startup flags, shell-style string, e.g. "--attention sdpa --profile 4"
    WAN2GP_SERVER_HOST          Bind host (default: 0.0.0.0)
    WAN2GP_SERVER_PORT          Bind port (default: 8000)
    WAN2GP_SERVER_OUTPUT_DIR    Override WanGP's output folder (default: WanGP config default)
    WAN2GP_SERVER_DATA_DIR      Where uploaded/downloaded input images are stored
                                (default: <wan2gp_server>/data)
    WAN2GP_SERVER_T2I_MODEL     Default text-to-image preset id   (default: z-image-turbo)
    WAN2GP_SERVER_T2V_MODEL     Default text-to-video preset id   (default: wan21-t2v-1.3b)
    WAN2GP_SERVER_I2V_MODEL     Default image-to-video preset id  (default: wan21-fun-inp-1.3b)
    WAN2GP_SERVER_PRESETS_DIR   Optional folder with extra *.json model presets
    WAN2GP_SERVER_EAGER_INIT    "1" to load the WanGP runtime at startup instead of on first job
    WAN2GP_SERVER_API_KEY       If set, all /v1 requests must send it in the X-API-Key header
"""

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_SERVER_DIR = Path(__file__).resolve().parent
# wan2gp_server/ lives inside the Wan2GP repository by default
_DEFAULT_WAN2GP_ROOT = _SERVER_DIR.parent


@dataclass
class ServerConfig:
    wan2gp_root: Path = _DEFAULT_WAN2GP_ROOT
    cli_args: List[str] = field(default_factory=list)
    host: str = "0.0.0.0"
    port: int = 8000
    output_dir: Optional[Path] = None
    data_dir: Path = _SERVER_DIR / "data"
    default_models: Dict[str, str] = field(
        default_factory=lambda: {
            # T4-friendly defaults (free Colab); override via env for bigger GPUs
            "t2i": "z-image-turbo",
            "t2v": "wan21-t2v-1.3b",
            "i2v": "wan21-fun-inp-1.3b",
        }
    )
    presets_dir: Optional[Path] = None
    eager_init: bool = False
    api_key: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ServerConfig":
        cfg = cls()

        if os.environ.get("WAN2GP_ROOT"):
            cfg.wan2gp_root = Path(os.environ["WAN2GP_ROOT"]).resolve()
        if os.environ.get("WAN2GP_CLI_ARGS"):
            cfg.cli_args = shlex.split(os.environ["WAN2GP_CLI_ARGS"])
        cfg.host = os.environ.get("WAN2GP_SERVER_HOST", cfg.host)
        cfg.port = int(os.environ.get("WAN2GP_SERVER_PORT", cfg.port))
        if os.environ.get("WAN2GP_SERVER_OUTPUT_DIR"):
            cfg.output_dir = Path(os.environ["WAN2GP_SERVER_OUTPUT_DIR"]).resolve()
        if os.environ.get("WAN2GP_SERVER_DATA_DIR"):
            cfg.data_dir = Path(os.environ["WAN2GP_SERVER_DATA_DIR"]).resolve()
        for task, var in (
            ("t2i", "WAN2GP_SERVER_T2I_MODEL"),
            ("t2v", "WAN2GP_SERVER_T2V_MODEL"),
            ("i2v", "WAN2GP_SERVER_I2V_MODEL"),
        ):
            if os.environ.get(var):
                cfg.default_models[task] = os.environ[var]
        if os.environ.get("WAN2GP_SERVER_PRESETS_DIR"):
            cfg.presets_dir = Path(os.environ["WAN2GP_SERVER_PRESETS_DIR"]).resolve()
        cfg.eager_init = os.environ.get("WAN2GP_SERVER_EAGER_INIT", "0") == "1"
        cfg.api_key = os.environ.get("WAN2GP_SERVER_API_KEY") or None

        return cfg

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"
