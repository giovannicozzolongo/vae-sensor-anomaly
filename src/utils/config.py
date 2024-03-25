import yaml
from pathlib import Path


def load_config(path: str | None = None) -> dict:
    """Load experiment config from YAML file."""
    if path is None:
        path = (
            Path(__file__).parent.parent.parent / "configs" / "experiment_config.yaml"
        )
    else:
        path = Path(path)

    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg
