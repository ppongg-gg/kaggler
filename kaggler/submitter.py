import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import GlobalConfig

logger = logging.getLogger(__name__)


@dataclass
class SubmissionRecord:
    iteration: int
    timestamp: str
    filename: str
    submission_message: str
    preset_used: str
    val_score: float
    public_lb_score: Optional[float] = None
    public_lb_rank: Optional[int] = None
    status: str = "pending"
    training_duration_seconds: float = 0.0
    strategy_applied: Optional[str] = None


class Submitter:
    def __init__(self, config: GlobalConfig, competition_name: str, workspace: Path):
        self.config = config
        self.competition_name = competition_name
        self.log_path = workspace / "submissions" / "submission_log.json"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def submit(
        self,
        submission_path: Path,
        iteration: int,
        message: str,
        preset_used: str = "",
        val_score: float = 0.0,
        training_duration: float = 0.0,
        strategy_applied: Optional[str] = None,
        skip_submit: bool = False,
    ) -> SubmissionRecord:
        record = SubmissionRecord(
            iteration=iteration,
            timestamp=datetime.now(timezone.utc).isoformat(),
            filename=submission_path.name,
            submission_message=message,
            preset_used=preset_used,
            val_score=val_score,
            training_duration_seconds=training_duration,
            strategy_applied=strategy_applied,
        )

        if skip_submit:
            logger.info(f"--skip-submit: not uploading {submission_path.name}")
            record.status = "skipped"
            self._append_record(record)
            return record

        used, remaining = self.check_daily_budget()
        if remaining <= 0:
            logger.warning("Daily submission budget exhausted. Skipping submission.")
            record.status = "budget_exhausted"
            self._append_record(record)
            return record

        import kaggle
        logger.info(f"Submitting iteration {iteration}: {submission_path.name}")
        try:
            kaggle.api.competition_submit(
                file_name=str(submission_path),
                message=message,
                competition=self.competition_name,
            )
            record.status = "submitted"
        except Exception as e:
            logger.error(f"Submission failed: {e}")
            record.status = "error"
            self._append_record(record)
            return record

        self._append_record(record)

        # Poll for score
        score = self._poll_for_score(timeout=300)
        if score is not None:
            record.public_lb_score = score
            record.status = "complete"
            self._update_record(record)
            logger.info(f"Public LB score: {score:.5f}")
        else:
            logger.info("Score not available yet (will remain as 'submitted' in log).")

        return record

    def check_daily_budget(self) -> tuple:
        today = datetime.now(timezone.utc).date().isoformat()
        records = self._load_records()
        submitted_today = sum(
            1 for r in records
            if r.get("timestamp", "").startswith(today)
            and r.get("status") not in ("skipped", "error", "budget_exhausted")
        )

        # Cross-check with Kaggle API
        try:
            import kaggle
            api_subs = kaggle.api.competition_submissions(self.competition_name)
            api_today = sum(
                1 for s in api_subs
                if str(getattr(s, "date", "")).startswith(today)
            )
            submitted_today = max(submitted_today, api_today)
        except Exception as e:
            logger.debug(f"Could not check Kaggle API submissions: {e}")

        remaining = max(0, self.config.max_daily_submissions - submitted_today)
        return submitted_today, remaining

    def _poll_for_score(self, timeout: int = 300) -> Optional[float]:
        import kaggle
        deadline = time.time() + timeout
        while time.time() < deadline:
            time.sleep(30)
            try:
                subs = kaggle.api.competition_submissions(self.competition_name)
                if subs:
                    latest = subs[0]
                    status = str(getattr(latest, "status", "")).lower()
                    if status == "complete":
                        score = getattr(latest, "publicScore", None)
                        return float(score) if score is not None else None
                    elif status == "error":
                        logger.warning("Submission returned error status from Kaggle.")
                        return None
            except Exception as e:
                logger.debug(f"Score polling error: {e}")
        return None

    def _load_records(self) -> list:
        if self.log_path.exists():
            data = json.loads(self.log_path.read_text())
            return data.get("submissions", [])
        return []

    def _append_record(self, record: SubmissionRecord) -> None:
        data = {"competition": self.competition_name, "submissions": self._load_records()}
        data["submissions"].append(asdict(record))
        self.log_path.write_text(json.dumps(data, indent=2))

    def _update_record(self, record: SubmissionRecord) -> None:
        data = {"competition": self.competition_name, "submissions": self._load_records()}
        for i, r in enumerate(data["submissions"]):
            if r.get("iteration") == record.iteration:
                data["submissions"][i] = asdict(record)
                break
        self.log_path.write_text(json.dumps(data, indent=2))

    def get_all_records(self) -> list:
        return self._load_records()
