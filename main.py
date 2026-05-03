import argparse
import logging
import sys

from kaggler.config import ConfigManager
from kaggler.orchestrator import KaggleOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated Kaggle Bot — reads competition, trains with AutoGluon, submits and iterates."
    )
    parser.add_argument("--competition", required=True, help="Kaggle competition slug (e.g. titanic)")
    parser.add_argument("--iterations", type=int, default=5, help="Max iterations to run (default: 5)")
    parser.add_argument("--skip-submit", action="store_true", help="Train and predict but don't submit to Kaggle")
    parser.add_argument("--reset", action="store_true", help="Clear models/submissions (keeps raw data) and start fresh")
    parser.add_argument("--verbose", action="store_true", help="Show debug-level logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    config_manager = ConfigManager(args.competition)

    if args.reset:
        try:
            global_cfg = config_manager.load_global()
            workspace = config_manager.get_workspace(global_cfg)
            config_manager.clear_workspace(workspace)
            print(f"Workspace cleared for '{args.competition}'.")
        except EnvironmentError as e:
            print(f"Warning: {e}")

    orchestrator = KaggleOrchestrator(
        competition_name=args.competition,
        max_iterations=args.iterations,
        config_manager=config_manager,
        skip_submit=args.skip_submit,
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
