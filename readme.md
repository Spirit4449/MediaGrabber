# MediaGrabber - Telegram Media Sync System

**A secure, production-grade Telegram media synchronization and downloader** with three modes:

1. **On-Demand Downloads** - POST `/api/download` with HMAC-signed requests to download any Telegram post
2. **Scheduled Daily Sync** - Automatically monitors a source channel and uploads new media to target channel (runs daily at 12 PM via PM2)
3. **Bot Commands** - Telegraf bot interface for interactive use

**Features:**

- ✅ Downloads media from public channels (t.me/username/post) and private channels (t.me/c/id/post)
- ✅ Streams progress to Telegram chat
- ✅ Auto-restarts if server crashes (via PM2)
- ✅ Rate-limited API with HMAC-SHA256 authentication
- ✅ No static file serving (security hardened)
- ✅ Stateful daily sync (tracks processed messages)
- ✅ Absolute path support for cross-machine deployment

**Designed for:**

- Linux/Mac servers and Raspberry Pi
- 24/7 operation with PM2 process manager
- Nginx reverse proxy (sample config included)

## Architecture

```
Telegram (Public & Private Channels)
    ↓
    ├─→ [On-Demand API] server.js + downloader.py
    │   POST /api/download with HMAC signature
    │   Downloads immediately, uploads to chat
    │
    └─→ [Daily Scheduler] daily_bns_sync.py via PM2
        Runs at 12 PM daily
        Monitors source channel, syncs new media
        Maintains .state/bns_state.json
```

**Three Components:**

| Component             | Type    | Purpose                    | Runs                             |
| --------------------- | ------- | -------------------------- | -------------------------------- |
| **server.js**         | Node.js | Express API + Telegraf bot | 24/7 (PM2 managed)               |
| **downloader.py**     | Python  | Media download worker      | On-demand (spawned by server.js) |
| **daily_bns_sync.py** | Python  | Daily sync monitor         | Daily at 12 PM (PM2 cron)        |

**Tech Stack:**

- **Telegraf** - Telegram Bot API wrapper (Node.js)
- **Telethon** - Telegram user account client (Python 3)
- **Express.js** - HTTP API server
- **PM2** - Process manager + scheduler
- **dotenv** - Environment variable loader
- **HMAC-SHA256** - API request authentication

## Requirements

**System:**

- Linux/Mac (or Raspberry Pi) or Windows with WSL2
- Node.js 18+ (`node --version`)
- Python 3.10+ (`python3 --version`)

**Credentials (from Telegram):**

1. **Bot Token** - Create bot via [@BotFather](https://t.me/botfather) → Get token
2. **User API Credentials** - Register app at [my.telegram.org](https://my.telegram.org) → Get API ID & Hash

**Optional:**

- Nginx (for HTTPS reverse proxy)
- Google Gemini API key (if using `telegram_scraper.py`)
- PM2 globally installed (`npm install -g pm2`) - for persistent management

## Setup (Linux/Mac)

### 1. Clone/Extract and Install Dependencies

```bash
cd MediaGrabber

# Install Node dependencies
npm install

# Create Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
```

Edit `.env`:

```ini
# Express server
PORT=4000
BIND_HOST=127.0.0.1
SHARED_SECRET=your-random-secret-here-64-chars-min

# Telegram Bot (from @BotFather)
BOT_TOKEN=123456789:ABCDEFghijklmnopqrstuvwxyz

# Telegram User API (from my.telegram.org)
TELEGRAM_API_ID=27434653
TELEGRAM_API_HASH=41a4175c8c019f5b61b6484af9d02c64

# Python binary path
PYTHON_BIN=python
```

### 3. First Run - Telethon Login

**Important:** Telethon requires user account login to create a session. This is one-time setup.

```bash
# Activate venv
source .venv/bin/activate

# Run downloader with --preflight (triggers login)
export $(grep -v '^#' .env | xargs)
python downloader.py --link https://t.me/example/123 --preflight
```

You'll be prompted:

1. Enter your phone number
2. Enter the verification code sent to Telegram
3. Enter 2FA password (if enabled)

This creates `media_grabber_session.session` - **keep this file safe!**

### 4. Start Server

**Option A: Manual (development)**

```bash
npm start
```

**Option B: PM2 (production, recommended)**

First install PM2:

```bash
npm install -g pm2
```

Start with ecosystem config:

```bash
pm2 start ecosystem.config.cjs
pm2 save  # persist on reboot
```

View logs:

```bash
pm2 logs
pm2 logs satsangfetcher
pm2 logs bns_daily
```

## Setup (Windows)

**Warning:** Paths are different. In `ecosystem.config.cjs`, change:

```javascript
PYTHON_BIN: "./.venv/Scripts/python.exe",  // Windows path
// Also update cwd and TELEGRAM_SESSION paths
```

Or use **WSL2 (recommended)** - then follow Linux setup.

## Usage

### Mode 1: On-Demand Download via API

```bash
curl -X POST http://127.0.0.1:4000/api/download \
  -H "Content-Type: application/json" \
  -H "x-signature: <HMAC-SHA256>" \
  -d '{
    "link": "https://t.me/c/1773081661/4385",
    "chat_id": 123456789,
    "caption": "Downloaded media",
    "forceVideo": false
  }'
```

**Compute HMAC (Node.js):**

```javascript
const crypto = require("crypto");
const body = { link, chat_id, caption };
const sig = crypto
  .createHmac("sha256", process.env.SHARED_SECRET)
  .update(JSON.stringify(body))
  .digest("hex");
```

### Mode 2: Daily Scheduled Sync

**Automatic via PM2:**

```bash
pm2 start ecosystem.config.cjs
# daily_bns_sync.py runs automatically at 12:00 (12 PM) daily
```

**Manual trigger:**

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
python daily_bns_sync.py [--source 123 --target 456 --seed]
```

**Arguments:**

- `--source` - Source channel ID (default: 1773081661)
- `--target` - Target channel ID (default: 3130614830)
- `--outdir` - Download directory (default: downloads/Brahma\ naa\ sange/)
- `--seed` - Force process recent messages on first run (else just seeds state)

**State tracking:**

```
.state/
└── bns_state.json  # Stores last_processed_id to avoid re-downloading
```

### Mode 3: Manual Download Scripts

**One-off download:**

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)
python downloader.py --link https://t.me/c/123/456 --preflight
```

**Advanced scraping with AI categorization:**

```bash
python telegram_scraper.py config.json
# Requires Gemini API key + APScheduler
```

## PM2 - Process Management

The `ecosystem.config.cjs` configures two processes:

### Process 1: `satsangfetcher` (server.js)

```javascript
{
  name: "satsangfetcher",
  script: "server.js",
  instances: 1,
  autorestart: true,        // Auto-restart on crash
  watch: false,
  cron_restart: false,
  env: {
    NODE_ENV: "production",
    PYTHON_BIN: "./.venv/bin/python",
    BOT_TOKEN: "...",
    SHARED_SECRET: "...",
    PORT: "4000"
  }
}
```

**Commands:**

```bash
pm2 start ecosystem.config.cjs --only satsangfetcher
pm2 restart satsangfetcher
pm2 stop satsangfetcher
pm2 logs satsangfetcher
pm2 monit  # real-time stats
```

### Process 2: `bns_daily` (daily_bns_sync.py)

```javascript
{
  name: "bns_daily",
  script: "./daily_bns_sync.py",
  instances: 1,
  autorestart: false,         // Don't auto-restart on crash
  cron_restart: "0 17 * * *", // Run at 5 PM daily
  env: {
    TELEGRAM_API_ID: "...",
    TELEGRAM_API_HASH: "...",
  }
}
```

**Why separate processes?**

- Server runs 24/7
- Daily script runs on schedule
- Independent logging
- Can restart one without affecting the other

**View state:**

```bash
pm2 list
pm2 info bns_daily
```

## Security & Nginx

**Why Nginx?**

- Terminates HTTPS/SSL
- Hides internal IP (127.0.0.1)
- Acts as reverse proxy to localhost:4000
- Can limit request sizes & rates at network level

**Sample config (nginx.conf.sample):**
See the included `nginx.conf.sample` for HTTPS + rate limiting setup.

## Session Management

**Session file:** `media_grabber_session.session`

- Created after first Telethon login
- Contains authenticated session data
- **DO NOT commit to git** (already in .gitignore)
- **Must exist on each machine** - run `--preflight` once per host

**If changing API credentials:**

```bash
rm media_grabber_session.session
python downloader.py --link https://t.me/example/123 --preflight
# Re-authenticate
```

## Troubleshooting

| Error                                 | Solution                                          |
| ------------------------------------- | ------------------------------------------------- |
| `TELEGRAM_API_ID/HASH required`       | Add to `.env` or set via `export`                 |
| `Could not find the input entity`     | User account must join source channel first       |
| `No new messages to process`          | State already up-to-date or no new posts          |
| `Permission denied sending to target` | Verify bot is admin in target channel             |
| `Large files timeout`                 | Increase timeout in `daily_bns_sync.py`           |
| Server crashes                        | Check `pm2 logs` - verify credentials/permissions |

## What We Fixed ✅

**Problem:** Daily downloader (`daily_bns_sync.py`) failed with:

```
TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables are required
```

**Root Cause:** Script didn't load `.env` file (unlike Node.js bot which uses `dotenv`).

**Solution:** Added dotenv loading:

```python
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
```

**Result:** Daily sync now works automatically! 🎉

## File Structure

```
MediaGrabber/
├── server.js                    # Express + Telegraf bot (runs 24/7)
├── downloader.py               # Single-use downloader worker
├── daily_bns_sync.py            # Daily sync monitor (runs at 5 PM)
├── ecosystem.config.cjs         # PM2 configuration
├── package.json                 # Node dependencies
├── requirements.txt             # Python dependencies
├── .env                         # Your credentials (don't commit!)
├── .env.example                 # Template for .env
├── nginx.conf.sample            # Nginx reverse proxy config
├── media_grabber_session.session # Telethon auth session (created on first run)
│
├── Optional utilities:
│   ├── telegram_scraper.py      # Advanced scraper with Gemini AI
│   ├── genai.py                 # Gemini API test
│   ├── grabber.py               # Alternative downloader
│   └── box.js                   # Utility script
│
└── Generated on first run:
    ├── .state/bns_state.json    # Daily sync state tracker
    ├── logs/                    # PM2 logs
    ├── downloads/               # Downloaded media (temp storage)
    └── .venv/                   # Python virtual environment
```

## FAQ

**Q: Can I run this without PM2?**
A: Yes. For on-demand only:

```bash
npm start
```

Then manually schedule `daily_bns_sync.py` with `cron`, systemd timer, or Task Scheduler.

**Q: Does this work on Raspberry Pi?**
A: Yes! Perfect for it. Tested on RPi 4 with 4GB RAM running 24/7.

**Q: What if I want different channels?**
A: For daily sync:

```bash
python daily_bns_sync.py --source 123456 --target 789012
```

Or edit `daily_bns_sync.py` constants.

**Q: Is this production-ready?**
A: Yes, but:

- Use strong `SHARED_SECRET` (64+ characters)
- Keep `.env` and session file secure
- Monitor logs with `pm2 logs`
- Use HTTPS via Nginx for public APIs
- Test with `--preflight` after setup

**Q: Maximum file size?**
A: Telegram API limit ~2GB. Timeout after 20+ minutes of no progress.

**Q: Can I track the sync history?**
A: Yes, check `.state/bns_state.json` and `pm2 logs bns_daily`.

## Known Limitations

- **Paths:** `ecosystem.config.cjs` has hardcoded absolute paths - adjust for your system
- **Windows:** Requires WSL2 or manual path adjustments
- **Session:** Tied to Telegram account + API credentials - must re-auth if either changes
- **File cleanup:** Downloaded files deleted after upload - no permanent archive (use target channel for that)

---

## License

[Add license here]

## Support

For issues:

1. Check logs: `pm2 logs`
2. Verify credentials in `.env`
3. Ensure bot is admin in target channel
4. Ensure user account joined source channel
5. Try manual run with `--seed` to force processing

---

**Last Updated:** January 21, 2026  
**Status:** ✅ Daily Sync Working  
**Maintained by:** You!
