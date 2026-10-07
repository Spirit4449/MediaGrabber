# BNS Daily Sync Operator Guide

This guide covers how to run `daily_bns_sync.py`, how the state files work, what every flag does, and how to debug the Telegram-to-Brahma-Naa-Sange website flow.

## Quick Mental Model

`daily_bns_sync.py` does this:

1. Logs into Telegram with Telethon.
2. Reads the newest messages from the source channel.
3. Compares each message id against a state file containing `last_id`.
4. Processes messages with `id > last_id`.
5. For audio runs, converts audio to an MP3 under the configured upload size.
6. Uploads the MP3 to the BNS website with Selenium when site upload is enabled.
7. Reposts the media to the target Telegram channel.
8. Advances `last_id` after the message is handled.

The important state file for the audio website flow is:

```text
.state/bns_audio_state.json
```

It usually looks like:

```json
{"last_id": 5337}
```

To replay the latest full experience, set `last_id` to one before the source Telegram message you want to process, run the audio sync, then delete the duplicate target-channel Telegram message if needed.

## Common Commands

Run the normal audio flow manually:

```bash
cd /home/nisch/Desktop/MediaGrabber
.venv/bin/python daily_bns_sync.py \
  --target 3744810661 \
  --audio-only \
  --state-file .state/bns_audio_state.json
```

Replay source message `5337` through the full flow:

```bash
cd /home/nisch/Desktop/MediaGrabber
# First edit .state/bns_audio_state.json so last_id is 5336.
.venv/bin/python daily_bns_sync.py \
  --target 3744810661 \
  --audio-only \
  --state-file .state/bns_audio_state.json \
  --only-message-id 5337
```

Run the audio flow but skip the website upload:

```bash
cd /home/nisch/Desktop/MediaGrabber
.venv/bin/python daily_bns_sync.py \
  --target 3744810661 \
  --audio-only \
  --no-site-upload \
  --state-file .state/bns_audio_state.json
```

Run with extra website patience:

```bash
cd /home/nisch/Desktop/MediaGrabber
.venv/bin/python daily_bns_sync.py \
  --target 3744810661 \
  --audio-only \
  --state-file .state/bns_audio_state.json \
  --site-upload-timeout 7200 \
  --site-upload-stall-timeout 300
```

Show all script options:

```bash
cd /home/nisch/Desktop/MediaGrabber
.venv/bin/python daily_bns_sync.py --help
```

## Flags

`--source SOURCE`

Source Telegram channel id. Defaults to the built-in BNS source id.

`--target TARGET`

Target Telegram channel id. For your audio repost flow, use:

```text
3744810661
```

`--outdir OUTDIR`

Download directory for media. The current flow downloads to temporary folders during processing, so this is mostly legacy/default behavior.

`--seed`

When the state file does not exist, the script normally sets `last_id` to the newest source message and exits so it does not backfill hundreds of old messages. With `--seed`, it processes the recent window instead.

Use this carefully.

`--state-file STATE_FILE`

Path to the JSON file that stores `last_id`.

Common files:

```text
.state/bns_state.json
.state/bns_audio_state.json
```

`--audio-only`

Only process audio messages. This is also required for website upload to run.

`--session SESSION`

Override the Telegram session file. Useful if you want to test with a separate Telethon session.

Common session files:

```text
media_grabber_session.session
media_grabber_session_audio.session
```

`--site-upload-timeout SECONDS`

Maximum time Selenium waits for the BNS website processing queue to finish one upload.

The website upload can include transcription, analysis, embeddings, and saving to Supabase, so large files can take a long time. A practical value is:

```text
7200
```

`--site-upload-stall-timeout SECONDS`

How long Selenium waits after a browser/API quota-looking stall before reporting a failure. A practical value is:

```text
300
```

`--no-site-upload`

Skips the BNS website upload and only performs the Telegram repost flow.

`--only-message-id MESSAGE_ID`

Process only one specific source Telegram message from the recent source window. The message still must be newer than `last_id`, so set the state file back first when replaying a processed message.

Example:

```bash
.venv/bin/python daily_bns_sync.py \
  --target 3744810661 \
  --audio-only \
  --state-file .state/bns_audio_state.json \
  --only-message-id 5337
```

## Environment Variables

These are loaded from `.env` when `python-dotenv` is installed.

Required Telegram variables:

```text
TELEGRAM_API_ID
TELEGRAM_API_HASH
TELEGRAM_SESSION
```

Notification variables:

```text
BOT_TOKEN
TELEGRAM_NOTIFY_CHAT_ID
```

BNS website variables:

```text
BNS_SITE_URL
BNS_ADMIN_USERNAME
BNS_ADMIN_PASSWORD
BNS_SELENIUM_HEADLESS
BNS_CHROMEDRIVER_PATH
BNS_CHROME_BINARY
BNS_SELENIUM_DEBUG_DIR
BNS_UPLOAD_TIMEOUT_SECONDS
BNS_UPLOAD_STALL_SECONDS
```

Audio conversion variables:

```text
FFMPEG_PATH
FFPROBE_PATH
BNS_MAX_UPLOAD_MB
```

`BNS_MAX_UPLOAD_MB` defaults to `49`. The script converts audio to MP3 and compresses as needed to stay under this size.

## PM2 Scheduled Runs

The PM2 config is:

```text
ecosystem.config.cjs
```

Relevant apps:

```text
bns_daily
bns_daily_audio
```

`bns_daily_audio` runs daily at 08:00 and uses:

```bash
daily_bns_sync.py --target 3744810661 --audio-only --state-file .state/bns_audio_state.json
```

Useful PM2 commands:

```bash
cd /home/nisch/Desktop/MediaGrabber
pm2 status
pm2 logs bns_daily_audio
pm2 logs bns_daily
pm2 restart bns_daily_audio
pm2 save
```

If you changed `.env`, the next fresh script start reads it. A PM2 restart is useful when you want to force the next run with new environment/config immediately.

## Logs And Debug Artifacts

Daily audio logs:

```text
logs/bns_audio_out-8.log
logs/bns_audio_err-8.log
logs/bns_audio_out.log
logs/bns_audio_err.log
```

Other daily sync logs:

```text
logs/bns_out-7.log
logs/bns_err-7.log
logs/bns_out.log
logs/bns_err.log
```

Selenium artifacts:

```text
logs/selenium/
```

On website upload failures, the script writes an HTML snapshot and screenshot such as:

```text
logs/selenium/upload-timeout-YYYYMMDD-HHMMSS.html
logs/selenium/upload-timeout-YYYYMMDD-HHMMSS.png
```

Open the HTML or PNG to see exactly what Selenium saw.

## Website Upload Success Criteria

The script should not treat stale archive/calendar entries as success.

Current success means:

```text
The Processing Queue card for the exact file name reaches Archived / 100%.
```

While waiting, the script prints a progress line about once per minute:

```text
BNS website upload still running for 2026-07-08.mp3: ...
```

If it times out, the error includes:

```text
Timed out waiting for upload completion
Last queue status
HTML/Screenshot debug artifact paths
Browser console excerpt, when relevant
```

## Your Replay Workflow

This is your normal full-flow simulation pattern.

1. Find the source Telegram message id you want to replay.
2. Edit `.state/bns_audio_state.json`.
3. Set `last_id` to one less than that message id.
4. Run the normal audio command.
5. Watch logs and Telegram notifications.
6. Delete the duplicate target-channel Telegram post afterward if needed.

Example for message `5337`:

```json
{"last_id": 5336}
```

Then:

```bash
cd /home/nisch/Desktop/MediaGrabber
.venv/bin/python daily_bns_sync.py \
  --target 3744810661 \
  --audio-only \
  --state-file .state/bns_audio_state.json \
  --only-message-id 5337
```

If you want the most realistic automatic behavior, omit `--only-message-id` after setting `last_id` back. The script will process all newer messages in order.

## Troubleshooting

### It says no new messages

Check the state file:

```bash
cat .state/bns_audio_state.json
```

If `last_id` is already equal to or newer than the source message, lower it and rerun.

### It uploads to Telegram but not the website

Check whether website upload was disabled:

```bash
grep -n "BNS_DISABLE_SITE_UPLOAD\\|BNS_ADMIN\\|BNS_SITE_URL" .env
```

Make sure the run includes `--audio-only`; website upload is intentionally allowed only during audio-only runs.

Check Selenium debug files:

```bash
ls -lt logs/selenium | head
```

### Admin verification fails

The script resolves credentials from:

1. `BNS_ADMIN_USERNAME` and `BNS_ADMIN_PASSWORD` in MediaGrabber `.env`, unless they are still the old default.
2. The sibling website env file:

```text
/home/nisch/Desktop/Brahma-naa-sange/.env.local
```

If the website admin accounts change, update one of those files.

### Website upload hangs

Use a longer timeout:

```bash
.venv/bin/python daily_bns_sync.py \
  --target 3744810661 \
  --audio-only \
  --state-file .state/bns_audio_state.json \
  --site-upload-timeout 7200 \
  --site-upload-stall-timeout 300
```

Look for progress lines:

```bash
tail -f logs/bns_audio_out-8.log
```

### Chrome or Selenium fails to start

Check:

```bash
which chromium-browser chromium chromedriver
grep -n "BNS_CHROME_BINARY\\|BNS_CHROMEDRIVER_PATH" .env
```

The current `.env` should point Selenium at local Chromium and ChromeDriver.

### Audio conversion fails

Check ffmpeg:

```bash
which ffmpeg
which ffprobe
```

If needed, set:

```text
FFMPEG_PATH=/path/to/ffmpeg
FFPROBE_PATH=/path/to/ffprobe
```

### Telegram session/login errors

Use the same session file configured in `.env` or pass `--session`.

If Telethon asks for login interactively, complete the phone/code/2FA flow once. The `.session` file is what makes future runs automatic.

## Useful Inspection Commands

Watch audio sync logs:

```bash
tail -f logs/bns_audio_out-8.log logs/bns_audio_err-8.log
```

See current state:

```bash
cat .state/bns_audio_state.json
```

See newest Selenium artifacts:

```bash
ls -lt logs/selenium | head -20
```

Check available flags:

```bash
.venv/bin/python daily_bns_sync.py --help
```

Check script syntax:

```bash
.venv/bin/python -m py_compile daily_bns_sync.py
```

