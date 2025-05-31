from typing import Any, Dict

import yaml


class ConfigManager:
    """Manages application configuration loading and access."""

    def __init__(self, config_path: str = "configs/yolo_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Warning: Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {e}. Using defaults.")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails."""
        return {"detector": {}, "tracking": {}}

    def get_detector_config(self) -> Dict[str, Any]:
        """Get detector configuration."""
        return self.config.get("detector", {})

    def get_tracking_config(self) -> Dict[str, Any]:
        """Get tracking configuration."""
        return self.config.get("tracking", {})
