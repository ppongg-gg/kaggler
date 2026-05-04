import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


@dataclass
class GlobalConfig:
    kaggle_api_token: str
    anthropic_api_key: str
    workspace_root: Path
    max_daily_submissions: int
    claude_model: str


@dataclass
class CompetitionConfig:
    competition_name: str
    task_type: str = ""
    target_column: str = ""
    eval_metric: str = ""
    eval_metric_direction: str = "maximize"
    id_column: Optional[str] = None
    time_limit_per_iter: int = 300
    feature_engineering_hints: list = field(default_factory=list)
    custom_preprocessing: dict = field(default_factory=dict)


class ConfigManager:
    def __init__(self, competition_name: str):
        self.competition_name = competition_name

    def load_global(self) -> GlobalConfig:
        kaggle_token = os.environ.get("KAGGLE_API_TOKEN", "")
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        workspace_root = Path(os.environ.get("WORKSPACE_ROOT", "./workspaces"))
        max_subs = int(os.environ.get("KAGGLE_MAX_DAILY_SUBMISSIONS", "10"))
        claude_model = os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001")

        missing = []
        if not kaggle_token:
            missing.append("KAGGLE_API_TOKEN")
        if not anthropic_key:
            missing.append("ANTHROPIC_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Copy .env.example to .env and fill in your credentials."
            )

        # Ensure the token is available to the kaggle SDK
        os.environ["KAGGLE_API_TOKEN"] = kaggle_token

        return GlobalConfig(
            kaggle_api_token=kaggle_token,
            anthropic_api_key=anthropic_key,
            workspace_root=workspace_root,
            max_daily_submissions=max_subs,
            claude_model=claude_model,
        )

    def get_workspace(self, global_cfg: Optional[GlobalConfig] = None) -> Path:
        if global_cfg:
            root = global_cfg.workspace_root
        else:
            root = Path(os.environ.get("WORKSPACE_ROOT", "./workspaces"))
        workspace = root / self.competition_name
        workspace.mkdir(parents=True, exist_ok=True)
        return workspace

    def load_competition(self, workspace: Path) -> CompetitionConfig:
        config_path = workspace / "competition_config.json"
        if config_path.exists():
            data = json.loads(config_path.read_text())
            return CompetitionConfig(**data)
        return CompetitionConfig(competition_name=self.competition_name)

    def save_competition(self, cfg: CompetitionConfig, workspace: Path) -> None:
        config_path = workspace / "competition_config.json"
        data = asdict(cfg)
        config_path.write_text(json.dumps(data, indent=2))

    def clear_workspace(self, workspace: Path) -> None:
        import shutil
        for subdir in ("models", "submissions", "data/processed", "logs"):
            target = workspace / subdir
            if target.exists():
                shutil.rmtree(target)
