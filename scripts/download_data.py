"""Download the BRISC dataset configured in YAML or JSON."""
import argparse
from src.core.config import load_config, project_path
from src.core.paths import ROOT
from src.datasets.download import ensure_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=ROOT / "configs/dataset/brisc2025.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    config = config.get("dataset", config)
    root = ensure_dataset(project_path(config["root"]), auto_download=True,
                          kaggle_handle=config["kaggle_handle"])
    print(root)


if __name__ == "__main__":
    main()
