# Optional tools

These scripts are independent of the on-demand bot/API and daily sync. Run commands from the repository root so existing image, session, state, and config paths keep their meaning.

## One-time MP3 migration

Uses the core Python requirements plus FFmpeg:

```bash
.venv/bin/python tools/one_time_mp3_migrate.py --help
.venv/bin/python tools/one_time_mp3_migrate.py --dry-run --source 123456 --target 789012
```

Without `--dry-run`, the tool uploads converted media to the target channel. It loads `.env` and records progress in `.state/one_time_mp3_migrate_state.json`.

## Image experiments

Install their additional dependencies only when needed:

```bash
.venv/bin/pip install -r tools/requirements.txt
.venv/bin/python tools/ai.py
.venv/bin/python tools/telegram_scraper.py --run-once
```

`ai.py` interactively analyzes files in the root `images/` directory using `google.genai`; it reads credentials through dotenv/the SDK environment. It uploads each selected image to Gemini.

`telegram_scraper.py` uses `google.generativeai`, Pillow, and APScheduler. It reads **root `config.json`**, not a positional CLI config argument. Required fields are `api_id`, `api_hash`, `channels` (a list), and `start_date` (ISO date). Optional fields include `gemini_api_key`, `gemini_model`, `daily_run_time`, `max_concurrency`, `phone_number`, and `session_name`. Configure the model explicitly for your account. With no `--run-once`, it starts a scheduler and an immediate pipeline run.

Its root `results.json`, `progress.json`, `pending_images.json`, `data/`, and `logs/` paths are unchanged. These files and `config.json` are ignored because they contain generated data or credentials. The two image tools use different Gemini SDKs; they remain separate optional experiments pending consolidation. Their live integrations are not covered by the core test suite.
