# tg-media-bot

Secure Telegram media fetcher:
- POST /api/download with an HMAC-signed request
- Downloads a Telegram post's media (public or t.me/c/private)
- Streams progress to the bot chat
- Uploads the finished file to the Telegram chat (as document/video)
- Deletes the file from disk

No public file endpoints. Designed for Nginx proxy to 127.0.0.1:4000.

## Requirements
- Node 18+ (or Docker)
- Python 3.10+ with pip
- A Telegram **bot token**
- A Telegram **user** API (api_id/api_hash) for Telethon

## Setup (bare metal)

1) Copy env:
cp .env.example .env

Fill in:
- BOT_TOKEN (bot)
- TELEGRAM_API_ID / TELEGRAM_API_HASH (user API from https://my.telegram.org)
- SHARED_SECRET (any random string)

2) Install Node deps:
npm i


3) Create Python venv + install requirements:


python -m venv .venv
. .venv/bin/activate # on Windows: .venv\Scripts\activate
pip install -r requirements.txt


4) Run:


npm start


5) (First run only) Telethon will ask you to login (code + possibly 2FA) in the console to create a session.
   This session is stored in `media_grabber_session.session`.

## Usage

Your bot/backend should POST to:


POST http://127.0.0.1:4000/api/download

Headers:
Content-Type: application/json
x-signature: <HMAC-SHA256 of JSON body with SHARED_SECRET>

Body:
{
"link": "https://t.me/c/1773081661/4385
",
"chat_id": 123456789,
"caption": "Optional caption",
"forceVideo": false
}


### HMAC example (Node)
```js
const crypto = require('crypto');
const body = { link, chat_id, caption: 'Hello' };
const sig = crypto.createHmac('sha256', process.env.SHARED_SECRET)
  .update(JSON.stringify(body))
  .digest('hex');

await fetch('http://127.0.0.1:4000/api/download', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json', 'x-signature': sig },
  body: JSON.stringify(body)
});

Security notes

No static file serving at all.

Express binds to 127.0.0.1 by default; proxy with Nginx.

HMAC auth on /api/download.

Rate-limited endpoint.

Downloads stored under ./downloads/ and deleted after upload.

Nginx (sample)

See nginx.conf.sample and adjust for your domain/SSL.


On first run, attach to server container logs to complete Telethon login.

docker compose logs -f server

Env rotation & sessions

If you change TELEGRAM_API_ID/HASH or run on a new host, you’ll have to re-login for Telethon to create a new session.

Session file is media_grabber_session.session.

Troubleshooting

“Could not find the input entity…” → ensure the user account used by Telethon has joined the chat for t.me/c links.

Large files: the Python worker has no overall timeout, only a no-progress watchdog that scales with size.

If your bot must always send as video, set "forceVideo": true in POST body.


---

## .env.example

```ini
# Express / security
PORT=4000
BIND_HOST=127.0.0.1
SHARED_SECRET=change-me-very-random

# Bot
BOT_TOKEN=123456:ABCDEF-your-bot-token

# Python
PYTHON_BIN=python

# Telethon user API (from my.telegram.org)
TELEGRAM_API_ID=123456
TELEGRAM_API_HASH=your_api_hash_here