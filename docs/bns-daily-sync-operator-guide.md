# BNS Daily Sync Operator Guide

This guide covers how to run `daily_bns_sync.py`, how the state files work, what every flag does, and how to debug the Telegram-to-Brahma-Naa-Sange website flow.

## Quick Mental Model

`daily_bns_sync.py` does this:

1. Logs into Telegram with Telethon.
2. Reads the newest messages from the source channel.
3. Compares each message id against a state file containing `last_id`.
4. Processes messages with `id > last_id`.
5. For audio runs, converts audio to an MP3 under the configured upload size.
6. When site upload is enabled, signs in to the BNS website over HTTP (`bns_site.py`), uploads the MP3 straight to the site's audio storage through a presigned URL, and queues it as a server-side job; the script waits for that job to finish. No browser is involved.
7. Reposts the media to the target Telegram channel.
8. Saves each completed delivery step and advances `last_id` only after required deliveries succeed. On failure it stops and retains the message for retry.

State is written atomically and locked against overlapping runs. Existing `last_id` files remain compatible. Corrupt state causes an error instead of silently reseeding. Messages after the checkpoint are fetched oldest first in batches of 200, even after long outages.

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
  --site-upload-timeout 7200
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

Maximum time the script waits for the BNS website's server job to finish one upload.

The job includes transcription, analysis, embeddings, and saving to Supabase, so large files can take a long time. If the wait times out, the job keeps running on the site and the next run attaches to it instead of uploading again. A practical value is:

```text
7200
```

`--no-site-upload`

Skips the BNS website upload and only performs the Telegram repost flow.

`--only-message-id MESSAGE_ID`

Fetch and process one specific source Telegram message by ID. The message still must be newer than `last_id`, so set the state file back first when replaying a processed message.

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
BNS_UPLOAD_TIMEOUT_SECONDS
```

Audio conversion variables:

```text
FFMPEG_PATH
FFPROBE_PATH
BNS_MAX_UPLOAD_MB
```

`BNS_MAX_UPLOAD_MB` defaults to `49`. The script converts audio to MP3 and compresses as needed to stay under this size.

## PM2 Recovery Workers

Follow the [Linux deployment and migration guide](recovery-deployment.md) to activate the new configuration. No new dependencies are required.

`bns_daily` and `bns_daily_audio` now run persistent `service_worker.py` processes. They launch bounded sync attempts every 15 minutes from noon and 08:00 respectively, in `BNS_TIMEZONE` (default `America/New_York`). Failed attempts retry, with a 30-minute limit per attempt; completed website steps are not repeated on a Telegram retry. Checks stop after today’s successful delivery or at 18:00 and resume the next day. At `BNS_ALERT_AFTER` (default `18:00`), missing delivery sends an alert to the existing personal notification chat. An in-flight attempt may finish within its 30-minute limit after the cutoff.

```bash
pm2 status
pm2 logs bns_daily_audio
pm2 restart bns_daily_audio
```

Stop the relevant worker before manual replay or editing its checkpoint, then restart it afterward. Back up state first. A targeted `--only-message-id` replay intentionally advances the checkpoint to that ID, so use a separate replay state file if earlier pending work must remain untouched. The normal script's per-state lock prevents concurrent attempts, but does not protect manual file edits.

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

Website upload failures are reported with the site's own error text (HTTP status and message) in the
log and in the Telegram alert.

## Website Upload Success Criteria

The script reads the signed-in account's jobs from the site's `/api/jobs` endpoint (with its signed-in session) and matches on the exact file name.

Current success means:

```text
The newest job for the exact file name has status "completed" (message "Archived").
```

Before uploading, the script checks for an existing job with that file name:

- `completed`: reported as uploaded, nothing is re-uploaded.
- still running: the script waits for it.
- `error` or `stopped`: the script presses the job's retry endpoint, which resumes from the site's saved checkpoints, then waits.

Jobs are listed per account, so keep `BNS_ADMIN_USERNAME` stable. Clearing the Uploads list on the site hides jobs from this check.

While waiting, the script prints a progress line about once per minute:

```text
BNS website upload still running for 2026-07-08.mp3: ...
```

If it times out, the error includes:

```text
Timed out waiting for upload completion
Last job status
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

### Sign-in fails

The script signs in with `POST /api/admin/login` and reports the site's own error text (for example `Invalid username or password.`). The site locks an IP out after 10 failed sign-ins in 15 minutes.

The script resolves credentials from:

1. `BNS_ADMIN_USERNAME` and `BNS_ADMIN_PASSWORD` in MediaGrabber `.env`, unless they are still the old default.
2. The first `ADMIN_CREDENTIALS` entry in the sibling website env file:

```text
/home/nisch/Desktop/Brahma-naa-sange/.env.local
```

The account must be in the website's `ADMIN_CREDENTIALS`. A `VIEWER_CREDENTIALS` account is refused with `is not an admin`.

### Website upload hangs

Use a longer timeout:

```bash
.venv/bin/python daily_bns_sync.py \
  --target 3744810661 \
  --audio-only \
  --state-file .state/bns_audio_state.json \
  --site-upload-timeout 7200
```

Look for progress lines:

```bash
tail -f logs/bns_audio_out-8.log
```

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

Automated sync never prompts for login. Provision or repair the saved session using `downloader.py --login` on the deployment host. BNS loads its authorization into memory so concurrent workers do not write to the same session database.

## Useful Inspection Commands

Watch audio sync logs:

```bash
tail -f logs/bns_audio_out-8.log logs/bns_audio_err-8.log
```

See current state:

```bash
cat .state/bns_audio_state.json
```

Check available flags:

```bash
.venv/bin/python daily_bns_sync.py --help
```

Check script syntax:

```bash
.venv/bin/python -m py_compile daily_bns_sync.py
```


## Retry limits

A remote upload can succeed just before a crash prevents saving its checkpoint. A retry can duplicate that Telegram post; website retries use the site's existing duplicate detection. Previously skipped posts are not rediscovered automatically—use targeted replay. A permanently failing message blocks later messages until resolved. Bot requests are not a durable queue and may need resending after a restart.

## Moments with Mahant Swami (`moments_sync.py`)

A separate, text-only import: the daily post on channel 2074676830 goes to the BNS
site's `/api/moments/import`, which turns it into a prasang. No media is downloaded and
no Telegram message is reposted.

- Every run reads the whole channel and sends every post with text. The site ignores
  unchanged posts, re-processes edited ones, and marks posts missing from the read as
  deleted. Repeats and retries are safe.
- A run then waits up to `MOMENTS_WAIT_SECONDS` (600) for today's post to finish on the
  site and sends a notification with its title and whether it was linked to a
  recording. If it is not finished, the run exits non-zero and the worker checks again.
- PM2 app `moments_daily` (`service_worker.py moments --start-at 08:20`) runs it every
  15 minutes from 08:20 in `BNS_TIMEZONE` until today's post is in, with the usual 18:00
  missing-delivery alert. State: `.state/moments_state.json`.
- It signs in with the same `BNS_SITE_URL` admin login as the audio upload, over plain
  HTTP requests (no Selenium). `BNS_DISABLE_SITE_UPLOAD` does not apply to it.

```bash
.venv/bin/python moments_sync.py --dry-run --out /tmp/moments.jsonl   # read only
.venv/bin/python moments_sync.py --limit 10 --state-file .state/moments_pilot.json   # pilot
.venv/bin/python moments_sync.py                                       # last 14 days (what the daily run does)
.venv/bin/python moments_sync.py --full                                # whole channel: backfill, notices deletions
pm2 logs moments_daily
```

The daily run reads only the last `MOMENTS_DAYS` (14) days of the channel, so older
edits or deletions are picked up only by a `--full` run. `--limit` and `--days` never mark
posts deleted, because they are not full reads. The BNS side is
described in the website repository's `docs/moments.md`.
