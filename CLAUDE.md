# Kaggle Bot

Automated Kaggle competition bot: reads competition descriptions with Claude API, trains ML models with AutoGluon, submits predictions, and iterates to improve scores.

## Setup

```bash
uv sync                # create venv and install all dependencies
cp .env.example .env   # fill in credentials
```

Credentials needed in `.env`:
- `KAGGLE_API_TOKEN` — from https://www.kaggle.com/settings/api (copy the `KGAT_...` token)
- `ANTHROPIC_API_KEY` — from https://console.anthropic.com

## Usage

```bash
# Run 5 iterations on Titanic (fully automatic)
uv run python main.py --competition titanic --iterations 5

# Dry run — train and generate predictions but don't submit
uv run python main.py --competition titanic --iterations 1 --skip-submit

# Start fresh (clears models/submissions, keeps raw data)
uv run python main.py --competition titanic --reset --iterations 5
```

## Adding / updating dependencies

```bash
uv add <package>          # add a runtime dependency
uv add --dev <package>    # add a dev dependency
uv sync                   # re-sync the venv after editing pyproject.toml
```

## Architecture

```
main.py → KaggleOrchestrator
  ├── CompetitionReader   kaggler/competition_reader.py   Claude API → CompetitionMeta JSON
  ├── DataManager         kaggler/data_manager.py         Kaggle API download + inventory
  └── Iteration loop
        ├── Preprocessor  kaggler/preprocessor.py         Drop IDs, parse datetimes
        ├── Trainer       kaggler/trainer.py              AutoGluon TabularPredictor
        ├── Predictor     kaggler/predictor.py            Generate submission CSV
        ├── Submitter     kaggler/submitter.py            Kaggle submit + rate-limit guard
        ├── ScoreTracker  kaggler/tracker.py              Log scores per iteration
        └── StrategyAdvisor kaggler/strategy.py           Claude API → next strategy
```

## Workspace layout (auto-created, git-ignored)

```
workspaces/{competition}/
  competition_meta.json      # Claude-parsed competition info (hand-editable)
  competition_config.json    # Config overrides
  data/raw/                  # Downloaded files (never deleted by --reset)
  data/processed/            # Cleaned data
  models/iteration_{n}/      # AutoGluon artifacts
  submissions/iteration_{n}.csv
  submissions/submission_log.json
  logs/
```

## AutoGluon preset escalation

| Iteration | Preset         | Time limit |
|-----------|----------------|------------|
| 1         | medium_quality | 5 min      |
| 2         | good_quality   | 15 min     |
| 3         | high_quality   | 30 min     |
| 4         | best_quality   | 60 min     |
| 5+        | best_quality   | +30 min each |

## Key design decisions

- **AutoGluon handles most preprocessing** — the Preprocessor layer is intentionally thin (only drops ID cols and parses datetimes). Don't add encoding/imputation here.
- **competition_meta.json is cached** — delete it to force a re-read from Claude.
- **Daily submission budget** — checked against both local log and Kaggle API to catch manual submissions. Hard stop at limit.
- **Tabular only** — first version targets structured CSV competitions. NLP/CV not supported.

## Running tests

```bash
uv run pytest tests/
```
