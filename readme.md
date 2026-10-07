# MediaGrabber

MediaGrabber downloads Telegram posts on demand through a bot or an HMAC-authenticated HTTP API. A separate scheduled Python job copies channel media to another Telegram channel and can upload audio to the BNS website.

## Project layout

```text
server.js                      Node startup and service wiring
src/
  api.js                       HTTP routes, HMAC authentication, rate limiting
  bot.js                       Bot commands and invite conversation
  download-flow.js              Worker lifecycle and progress/delivery orchestration
  downloader-events.js         Ordered JSON-lines events from Python
  media-delivery.js             Telegram uploads and original download links
  media-compression.js          FFmpeg/FFprobe conversion
  shared-downloads.js           Publishing originals and retention cleanup
  progress.js                  Progress display formatting
  telegram-links.js            Post/invite link validation
downloader.py                  On-demand Python worker and login CLI
download_session.py            Read-only saved login → per-worker memory session
parallel_download.py           Bounded parallel document transfers
daily_bns_sync.py               Scheduled Telegram/BNS sync entry point
tools/                         Optional image tools and one-time migration
test/                          Node and Python regression tests
docs/                          Architecture review and deployment/operator guides
```

`server.js`, `downloader.py`, and `daily_bns_sync.py` remain at the root so existing PM2 entry points still work. The Node app resolves its `.env`, worker, and default download directories relative to the project root. Run Python commands from this directory; their relative session/state paths use the working directory.

Local `.env`, `.state/`, `logs/`, `downloads/`, `shared-downloads/`, and `*.session*` files are runtime data and are ignored by Git. Back up session and state files separately; sync checkpoints must survive deployment.

Before applying this cleanup to another checkout, back up its session files and `.state/` outside the repository. Git can remove the formerly tracked copies when updating that checkout; restore them before restarting services. This cleanup kept all local runtime files intact.

## Setup

Requires Node.js 18+ and Python 3.10+. Install FFmpeg and FFprobe for audio conversion or optional compressed copies; BNS website uploads also require Chrome/Chromium and a compatible driver.

```bash
npm ci
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Fill in the bot token, Telegram API ID/hash, and a long random `SHARED_SECRET`. The template uses `.venv/bin/python` for workers. Configure `PUBLIC_DOWNLOAD_BASE_URL` and Nginx if oversized originals should be shared; see [public downloads](docs/public-downloads.md).

Create the saved user login once, separately from active downloads:

```bash
.venv/bin/python downloader.py --login
```

The worker loads the project `.env` itself. Enter your phone number, Telegram code, and 2FA password if requested. This creates `media_grabber_session.session`, or the path named by `TELEGRAM_SESSION`. Workers read that login into independent in-memory sessions; they do not write to its SQLite database.

Start the bot and HTTP server:

```bash
npm start
```

For PM2, use your existing local config or create one from the template:

```bash
cp ecosystem.config.example.cjs ecosystem.config.cjs
# Review the schedule, timezone, channel IDs, and Python path first.
pm2 start ecosystem.config.cjs
pm2 save
```

The example schedules daily sync at 17:00 in the deployment host's timezone. PM2 starts the job immediately when first starting the config as well. Existing installations may use a different schedule or separate audio-sync process. The `.env` template disables website uploads until configured.

## On-demand downloads

Send a Telegram post URL to the bot, such as `https://t.me/channel/123` or `https://t.me/c/123456/789`. If channel access is missing, the bot asks for an invite. `/stop` and `/cancel` cancel the invite conversation; they do not cancel a running transfer.

HTTP clients call `POST /api/download` with JSON:

```json
{
  "link": "https://t.me/channel/123",
  "chat_id": 123456789,
  "caption": "Downloaded media",
  "forceVideo": false
}
```

Set `x-signature` to the hex HMAC-SHA256 of `JSON.stringify(body)`, using `SHARED_SECRET`:

```javascript
const signature = require('node:crypto')
  .createHmac('sha256', process.env.SHARED_SECRET)
  .update(JSON.stringify(body))
  .digest('hex');
```

The server signs the parsed/re-serialized JSON object, not the raw HTTP bytes. `{ "ok": true }` means the background download was started; completion or failure is delivered in Telegram. `GET /healthz` checks HTTP availability only.

Manual preflight or download:

```bash
.venv/bin/python downloader.py --link https://t.me/channel/123 --preflight
.venv/bin/python downloader.py --link https://t.me/channel/123
```

Files under the configured bot upload limit are sent to Telegram. Oversized originals are published unchanged with an expiring link; compressed copies are optional. Failed delivery retains the working original, but automatic delivery retries are not implemented.

### Download performance

Install `requirements.txt` in the environment used by `PYTHON_BIN`, then restart MediaGrabber. `cryptg` enables native Telegram decryption automatically. The bot checks access and downloads using one worker connection per request; cached/public channel lookups skip the full dialog list.

Documents of at least 8 MiB use four concurrent download streams. Set `DOWNLOAD_WORKERS=1` to use standard downloads, or choose 2–8 streams. Photos and smaller documents use standard downloads. Parallel failures retry with Telethon's standard downloader, and cancelled/failed downloads remove partial files. Progress updates are limited to every two seconds, with queued updates coalesced so delivery does not wait for a backlog of message edits.

The tests use simulated transfers; measure a real file on the deployment host to determine the speed improvement under its Telegram and network limits.

## Scheduled sync

```bash
.venv/bin/python daily_bns_sync.py --help
.venv/bin/python daily_bns_sync.py --source 123456 --target 789012
```

The daily script loads `.env` itself. With no checkpoint, its first run records the latest post and exits. `--seed` processes the recent window instead. Audio sync, per-run state/session files, website credentials, replay instructions, and known flags are covered in the [operator guide](docs/bns-daily-sync-operator-guide.md).

Current limitations: sync scans only the latest 200 messages; failures can be skipped if a later message advances the checkpoint; website and Telegram delivery are not tracked independently. See the [architecture review](docs/architecture.md) before relying on unattended catch-up or replay.

## Optional tools

The standalone scripts formerly at the root now live under `tools/`. Run them from the project root; update any external manual commands accordingly. They are not launched by the core server or the checked local PM2 config. The sample `images/` directory is used by `tools/ai.py` and has been retained.

See [tools setup and usage](tools/README.md). The duplicate interactive `grabber.py` has been removed; use `downloader.py --link ...` instead. The empty `box.js` was also removed.

## Verification and operations

```bash
npm test
.venv/bin/python -m unittest discover -s test -p 'test_*.py'
```

Node tests cover worker events, download orchestration, API routing/authentication, original delivery/retention, and real FFmpeg geometry/audio conversion. Encoding tests skip if FFmpeg/FFprobe are unavailable. Python tests cover saved sessions, concurrent transfers, cancellation, and fallback behavior. Tests do not contact Telegram or the BNS website.

- [Architecture and prioritized follow-ups](docs/architecture.md)
- [BNS daily sync operator guide](docs/bns-daily-sync-operator-guide.md)
- [Public download deployment](docs/public-downloads.md)
- [Domain migration](docs/domain-migration.md)
