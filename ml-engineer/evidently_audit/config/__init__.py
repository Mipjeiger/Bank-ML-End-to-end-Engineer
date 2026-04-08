from pathlib import Path
import yaml

def load_config(path=None) -> dict:
    cfg_path = Path(path) if path else Path(__file__).parent / "settings.yaml"
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)