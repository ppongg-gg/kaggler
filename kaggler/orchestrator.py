import logging
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from .config import ConfigManager, GlobalConfig
from .competition_reader import CompetitionReader, CompetitionMeta
from .data_manager import DataManager, DataInventory
from .preprocessor import Preprocessor
from .trainer import Trainer
from .predictor import Predictor
from .submitter import Submitter
from .tracker import ScoreTracker
from .strategy import StrategyAdvisor, StrategyAdvice

logger = logging.getLogger(__name__)
console = Console()


class KaggleOrchestrator:
    def __init__(
        self,
        competition_name: str,
        max_iterations: int,
        config_manager: ConfigManager,
        skip_submit: bool = False,
    ):
        self.competition_name = competition_name
        self.max_iterations = max_iterations
        self.config_manager = config_manager
        self.skip_submit = skip_submit

    def run(self) -> None:
        # --- Setup ---
        global_cfg = self.config_manager.load_global()
        workspace = self.config_manager.get_workspace(global_cfg)
        comp_cfg = self.config_manager.load_competition(workspace)

        console.print(Rule(f"[bold cyan]Kaggle Bot — {self.competition_name}"))

        # --- Phase 1: One-time setup (cached) ---
        reader = CompetitionReader(global_cfg)
        meta = reader.load_or_analyze(self.competition_name, workspace)
        console.print(
            f"[green]Competition:[/green] {meta.title}\n"
            f"[green]Task:[/green] {meta.task_type} | "
            f"[green]Metric:[/green] {meta.eval_metric} ({meta.eval_metric_direction})\n"
            f"[green]Target:[/green] {meta.target_column}"
        )

        # Sync comp_cfg from meta if fields are still empty
        if not comp_cfg.target_column:
            comp_cfg.target_column = meta.target_column
            comp_cfg.task_type = meta.task_type
            comp_cfg.eval_metric = meta.eval_metric
            comp_cfg.eval_metric_direction = meta.eval_metric_direction
            comp_cfg.id_column = meta.id_column
            self.config_manager.save_competition(comp_cfg, workspace)

        data_mgr = DataManager(self.competition_name, workspace)
        inventory = data_mgr.download()

        # Infer target column from sample_submission if meta didn't catch it
        if not meta.target_column or meta.target_column == "unknown":
            sample = data_mgr.load_sample_submission(inventory)
            if sample is not None:
                non_id_cols = [c for c in sample.columns if c.lower() != (meta.id_column or "id").lower()]
                if non_id_cols:
                    meta.target_column = non_id_cols[0]
                    logger.info(f"Inferred target column from sample_submission: {meta.target_column}")
                    import json, dataclasses
                    (workspace / "competition_meta.json").write_text(
                        json.dumps(dataclasses.asdict(meta), indent=2)
                    )

        # --- Phase 2: Iteration loop ---
        tracker = ScoreTracker(meta, workspace)
        submitter = Submitter(global_cfg, self.competition_name, workspace)
        advisor = StrategyAdvisor(global_cfg)
        strategy_advice: StrategyAdvice | None = None

        for iteration in range(1, self.max_iterations + 1):
            console.print(Rule(f"Iteration {iteration} / {self.max_iterations}"))

            # Check budget before doing any work
            used, remaining = submitter.check_daily_budget()
            if not self.skip_submit and remaining <= 0:
                console.print("[yellow]Daily submission limit reached. Stopping.[/yellow]")
                break

            # Preprocess
            preprocessor = Preprocessor(meta, inventory)
            extra_drops = strategy_advice.drop_features if strategy_advice else []
            extra_features = strategy_advice.feature_engineering_additions if strategy_advice else []
            preprocessor.build_plan(extra_drops=extra_drops, extra_features=extra_features)

            train_raw = data_mgr.load_train(inventory)
            test_raw = data_mgr.load_test(inventory)
            train_processed = preprocessor.fit_transform(train_raw)
            test_processed = preprocessor.transform(test_raw)

            # Train
            trainer = Trainer(meta, workspace)
            trainer_hints = strategy_advice.to_trainer_hints() if strategy_advice else None
            training_result = trainer.train(train_processed, iteration, trainer_hints)

            # Predict
            predictor = Predictor(meta)
            ag_predictor = trainer.load_predictor(iteration)
            sample_sub = data_mgr.load_sample_submission(inventory)
            submission_path = predictor.generate(
                ag_predictor, test_processed, sample_sub, iteration, workspace
            )

            # Submit
            strategy_desc = strategy_advice.summary if strategy_advice else None
            submission_record = submitter.submit(
                submission_path=submission_path,
                iteration=iteration,
                message=f"Iteration {iteration}: {training_result.preset_used}",
                preset_used=training_result.preset_used,
                val_score=training_result.val_score,
                training_duration=training_result.training_duration_seconds,
                strategy_applied=strategy_desc,
                skip_submit=self.skip_submit,
            )

            # Track
            tracker.record(training_result, submission_record)
            tracker.print_summary_table()

            # Strategize (skip on last iteration or if budget is nearly gone)
            is_last = iteration >= self.max_iterations
            if not is_last and (self.skip_submit or remaining > 1):
                strategy_advice = advisor.advise(
                    meta=meta,
                    history=tracker.get_history(),
                    autogluon_leaderboard=training_result.leaderboard,
                    remaining_submissions=max(0, remaining - 1),
                    remaining_iterations=self.max_iterations - iteration,
                )
                console.print(f"[dim]Strategy: {strategy_advice.summary}[/dim]")
                if strategy_advice.should_stop:
                    console.print(f"[yellow]Advisor recommends stopping: {strategy_advice.stop_reason}[/yellow]")
                    break

        # --- Phase 3: Final report ---
        self._print_final_report(tracker)

    def _print_final_report(self, tracker: ScoreTracker) -> None:
        console.print(Rule("[bold green]Final Report"))
        best = tracker.get_best()
        if best is None:
            console.print("[red]No iterations completed.[/red]")
            return

        lb_str = f"{best.public_lb_score:.5f}" if best.public_lb_score is not None else "N/A"
        console.print(
            f"[bold]Best iteration:[/bold] {best.iteration}\n"
            f"[bold]Public LB score:[/bold] {lb_str}\n"
            f"[bold]Validation score:[/bold] {best.val_score:.5f}\n"
            f"[bold]Preset used:[/bold] {best.preset_used}"
        )
        tracker.print_summary_table()
