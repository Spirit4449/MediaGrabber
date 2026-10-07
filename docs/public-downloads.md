# Public original-quality downloads

Oversized files are copied unchanged into a public directory and sent as a Telegram
button. The default is **link only**: there is no automatic compressed preview.
Small files continue to be sent to Telegram; non-MP3 audio is converted to MP3.
This applies to the interactive bot and `/api/download`; the separate daily sync
uses Telethon and is unchanged.

## Production environment

Add these values to production `.env` (or your PM2 environment):

```ini
PUBLIC_DOWNLOAD_BASE_URL=https://bns.classchats.net/downloads/
SHARED_DOWNLOAD_DIR=/var/www/mediagrabber-downloads
DOWNLOAD_RETENTION_HOURS=72
BOT_SEND_COMPRESSED_COPY=false
```

`PUBLIC_DOWNLOAD_BASE_URL` must include the public path Nginx serves. It is required
for oversized delivery; if absent, the bot reports a configuration error and keeps
the original in the working downloads directory. There is no Express static route.

Create the directory on the **production Linux device**, replacing `BOT_USER` and
`BOT_GROUP` with the user/group running PM2:

```sh
sudo install -d -m 0755 -o BOT_USER -g BOT_GROUP /var/www/mediagrabber-downloads
```

Published files are mode 0644; the directory and its parents must permit Nginx
traversal. A directory outside the repository avoids private home-directory
permissions blocking Nginx. SELinux-enabled hosts may also need the appropriate
web-readable file context.

## Nginx

Add this inside the existing HTTPS `server { ... }` for `bns.classchats.net`:

```nginx
location ^~ /downloads/ {
    alias /var/www/mediagrabber-downloads/;
    autoindex off;
    sendfile on;
    gzip off;
    add_header Content-Disposition "attachment" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Cache-Control "no-store" always;

    # Temporary staging files are never public.
    if ($uri ~ "(^|/)\.") { return 404; }

    # GET also permits HEAD. Nginx serves Range requests for download resume.
    limit_except GET { deny all; }
}
```

Both trailing slashes on the location and alias are intentional. This directory
contains public download files only, never `.env`, Telegram session files, logs,
or the repository. Names have a UUID prefix to prevent collisions. Anyone with a
URL can download. Nginx access logs can record these public URLs.

Validate and reload on production:

```sh
sudo nginx -t
sudo systemctl reload nginx
pm2 restart MediaGrabber --update-env
```

If dependencies need updating, run `npm ci` in the production repo first. No new
runtime npm packages were added.

## Retention and disk usage

The bot removes published files older than 72 hours at startup and every 15
minutes. Expiry survives restarts through file modification timestamps; no database
is required. Retention is approximate: a link remains usable until cleanup runs,
and cleanup cannot run while the bot is stopped. After deletion, new requests
return 404. Resuming a download requires the file to still exist.

Publication copies into a hidden temporary file and atomically renames it after
completion. The original is deleted only after Telegram accepts the link. Failed
publication or message delivery retains the working original. Failed-message
public copies expire normally. No disk quota is enforced: provision space for the
retained files plus in-flight downloads (briefly two copies when publishing).
Failed working originals are retained for manual recovery; review `downloads/`
periodically. Empty per-job download directories can also accumulate.

## Optional compression

To also send compressed copies of oversized video/audio, set:

```ini
BOT_SEND_COMPRESSED_COPY=true
BOT_COMPRESS_PRESET=medium
BOT_COMPRESS_TARGET_MB=47
BOT_COMPRESS_MIN_VIDEO_KBPS=300
BOT_COMPRESS_MAX_VIDEO_KBPS=2500
BOT_COMPRESS_AUDIO_KBPS=192
```

The link is sent before encoding. Video uses two-pass H.264, AAC audio when present,
Lanczos proportional resizing, square pixels, and rotation-aware dimensions. It
chooses a 720/480/360 short-edge ceiling based on bitrate and does not increase
input display dimensions. Output display ratio is checked within 1% (even-pixel
rounding). Impossible budgets leave the original link available instead of forcing
excessive quality loss. One retry is allowed for output exceeding the upload limit.
Jobs are serialized, use unique temporary directories, and limit video encoding
to two threads. Each FFmpeg pass has a 30-minute timeout. Audio converts directly
from the original, without an intermediate MP3 re-encode.

## Mac/Linux and verification

Use Node.js 18+ and FFmpeg/FFprobe on PATH, or set `FFMPEG_PATH` and `FFPROBE_PATH`.
FFmpeg must include `libx264`, `aac`, and `libmp3lame`. There are no Homebrew paths,
Linux `/dev/null` assumptions, or recently added FFmpeg scale options in the code.
The tests use generated media and Node's built-in test runner:

```sh
npm test
```

FFmpeg integration tests skip if FFmpeg/FFprobe are unavailable. Nginx installation,
permissions, HTTPS, and live Telegram delivery require verification on production:
send an oversized file, download it, compare checksums, test a Range request, and
confirm deletion using a short retention setting before returning it to 72 hours.
Existing PM2 absolute paths, Python virtualenv paths, and Telegram session locations
must match the production device. This change does not rewrite that deployment.
