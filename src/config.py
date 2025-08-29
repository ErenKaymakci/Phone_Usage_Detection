import yaml
from pathlib import Path

class Config:
    def __init__(self, config_file: str = "config.yaml"):
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r") as f:
            self.cfg = yaml.safe_load(f)

    @property
    def video_path(self) -> str:
        return self.cfg["video"]["path"]

    @property
    def model_path(self) -> str:
        return self.cfg["model"]["path"]

    @property
    def model_conf_threshold(self) -> float:
        return self.cfg["model"]["conf_threshold"]

    @property
    def model_iou_threshold(self) -> float:
        return self.cfg["model"]["iou_threshold"]

    @property
    def model_classes(self) -> list[str]:
        return self.cfg["model"].get("classes", [])

    @property
    def tracker_params(self) -> dict:
        return self.cfg["tracker"]

    @property
    def usage_rule_movement_threshold(self) -> int:
        return self.cfg["usage_rule"]["movement_threshold"]
