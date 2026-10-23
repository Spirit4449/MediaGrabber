#!/usr/bin/env python3
"""
daily_bns_sync.py

Fetch new media posts from a source channel and upload them to a target channel.

Default source: 1773081661
Default target: 3130614830

Behaviour:
- Maintains a small state file at .state/bns_state.json with last_id.
- On first run (no state), seeds the last_id to the latest message and exits to avoid backfilling.
  Use --seed to force processing of the recent window.

Requires TELEGRAM_API_ID, TELEGRAM_API_HASH and TELEGRAM_SESSION env variables (session file path).
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import shutil
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo
from bns_site import BnsSite
from download_session import load_download_session
from reliability import load_state, save_state, exclusive_lock
from pathlib import Path

from telethon import TelegramClient, types, utils as tutils  # type: ignore

# Load environment variables from .env file (for automated runs)
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on environment variables

# Minimal MIME -> ext fallback
MIME_EXT_FALLBACK = {
    "video/mp4": ".mp4",
    "video/x-matroska": ".mkv",
    "video/webm": ".webm",
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "audio/mpeg": ".mp3",
    "audio/ogg": ".ogg",
    "audio/x-m4a": ".m4a",
}

DEFAULT_SOURCE = 1773081661
DEFAULT_TARGET = 3130614830
STATE_DIR = Path(".state")
DEFAULT_STATE_FILE = STATE_DIR / "bns_state.json"
FFMPEG_PATH = os.environ.get("FFMPEG_PATH", "ffmpeg")
FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")
BNS_MAX_UPLOAD_MB = float(os.environ.get("BNS_MAX_UPLOAD_MB", "49"))
BNS_MAX_UPLOAD_BYTES = int(BNS_MAX_UPLOAD_MB * 1024 * 1024)
AUDIO_EXTS = {".mp3", ".m4a", ".ogg", ".oga", ".wav", ".flac", ".aac", ".opus", ".m4b"}
DEFAULT_BNS_ADMIN_USERNAME = "admin"
DEFAULT_BNS_ADMIN_PASSWORD = "admin123"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _clean_env_value(value: str) -> str:
    value = (value or "").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        env[key.strip()] = _clean_env_value(value)
    return env


def _parse_admin_credentials(raw: str) -> list[tuple[str, str]]:
    credentials: list[tuple[str, str]] = []
    for pair in _clean_env_value(raw).split(","):
        if ":" not in pair:
            continue
        username, password = pair.split(":", 1)
        username = username.strip()
        password = password.strip()
        if username and password:
            credentials.append((username, password))
    return credentials


def _load_bns_site_admin_credentials() -> tuple[str, str] | None:
    env_path = os.environ.get("BNS_SITE_ENV_FILE", "").strip()
    candidates = [Path(env_path)] if env_path else []
    candidates.extend(
        [
            Path(__file__).resolve().parent.parent / "Brahma-naa-sange" / ".env.local",
            Path.home() / "Desktop" / "Brahma-naa-sange" / ".env.local",
        ]
    )

    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)

        values = _read_env_file(resolved)
        for key in ("ADMIN_CREDENTIALS", "VITE_ADMIN_CREDENTIALS", "VITE_ADMIN_USERS"):
            parsed = _parse_admin_credentials(values.get(key, ""))
            if parsed:
                return parsed[0]

        username = values.get("VITE_ADMIN_USERNAME", "").strip()
        password = values.get("VITE_ADMIN_PASSWORD", "").strip()
        if username and password:
            return username, password
    return None


def _resolve_bns_admin_credentials() -> tuple[str, str]:
    username = os.environ.get("BNS_ADMIN_USERNAME", "").strip()
    password = os.environ.get("BNS_ADMIN_PASSWORD", "").strip()
    has_custom_env_credentials = bool(username and password) and (
        username != DEFAULT_BNS_ADMIN_USERNAME or password != DEFAULT_BNS_ADMIN_PASSWORD
    )
    if has_custom_env_credentials:
        return username, password

    site_credentials = _load_bns_site_admin_credentials()
    if site_credentials:
        return site_credentials

    return username, password


def _truncate(value: str, max_len: int = 1400) -> str:
    value = (value or "").strip()
    if len(value) <= max_len:
        return value
    return value[: max_len - 3].rstrip() + "..."


def send_telegram_bot_message(bot_token: str, chat_id: str, text: str) -> None:
    if not bot_token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = urllib.parse.urlencode(
        {
            "chat_id": chat_id,
            "text": _truncate(text, 3500),
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    req = urllib.request.Request(url, data=payload, method="POST")
    with urllib.request.urlopen(req, timeout=20) as res:
        if res.status >= 400:
            raise RuntimeError(f"Telegram Bot API HTTP {res.status}")


def notify(bot_token: str, chat_id: str, text: str) -> None:
    text = _truncate(text, 3500)
    print(text)
    if not bot_token or not chat_id:
        return
    try:
        send_telegram_bot_message(bot_token, chat_id, text)
    except Exception as exc:
        print(f"Telegram notify failed: {exc}")


def upload_to_bns_site(file_path: Path, site_url: str, admin_username: str, admin_password: str,
                       upload_timeout_seconds: int) -> dict:
    """Uploads one recording to the BNS site over HTTP and waits until it is processed (bns_site.py)."""
    try:
        site = BnsSite(site_url)
        site.login(admin_username, admin_password)
        return site.upload_recording(file_path, upload_timeout_seconds)
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "title": "", "summary": ""}


def _ext_from_media(msg) -> str:
    """Try to determine a sensible file extension for a message's media."""
    # Document (includes videos/voice/stickers sent as "file")
    if getattr(msg, "document", None):
        try:
            ext = tutils.get_extension(msg.document)
            if ext:
                return ext
        except Exception:
            pass
        mime = getattr(msg.document, "mime_type", None)
        return MIME_EXT_FALLBACK.get(mime, ".bin")
    # Photo
    if getattr(msg, "photo", None):
        return ".jpg"
    return ".bin"


def is_audio_message(msg) -> bool:
    """Return True if the message contains audio (including voice)."""
    if getattr(msg, "voice", None) or getattr(msg, "audio", None):
        return True
    doc = getattr(msg, "document", None)
    if doc and getattr(doc, "mime_type", "").startswith("audio/"):
        return True
    for attr in getattr(doc, "attributes", []) or []:
        if isinstance(attr, types.DocumentAttributeAudio):
            return True
    return False


def get_audio_duration_seconds(input_path: Path) -> float | None:
    if shutil.which(FFPROBE_PATH) is None:
        return None
    proc = subprocess.run(
        [
            FFPROBE_PATH,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=nk=1:nw=1",
            str(input_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if proc.returncode != 0:
        return None
    try:
        duration = float((proc.stdout or "").strip())
    except ValueError:
        return None
    return duration if duration > 0 else None


# Use encoder-supported rates and leave room for headers, tags, and frame padding.
MP3_BITRATES_KBPS = (192, 160, 128, 112, 96, 80, 64, 56, 48, 40, 32, 24, 16, 8)
MP3_UPLOAD_HEADROOM = 0.97


def mp3_bitrate_for_upload(input_path: Path) -> int | None:
    duration = get_audio_duration_seconds(input_path)
    if not duration:
        return None
    target_kbps = BNS_MAX_UPLOAD_BYTES * MP3_UPLOAD_HEADROOM * 8 / duration / 1000
    return next((rate for rate in MP3_BITRATES_KBPS if rate <= target_kbps), 8)


def convert_audio_to_mp3(input_path: Path) -> Path:
    """Convert audio to an MP3 that fits the measured upload limit."""
    if input_path.suffix.lower() == ".mp3" and input_path.stat().st_size <= BNS_MAX_UPLOAD_BYTES:
        return input_path
    if shutil.which(FFMPEG_PATH) is None:
        raise RuntimeError(f"ffmpeg not found: {FFMPEG_PATH}")

    output_path = input_path.with_suffix(".mp3")
    if output_path == input_path:
        output_path = input_path.with_name(f"{input_path.stem}-compressed.mp3")
    if output_path.exists():
        output_path.unlink()

    bitrate = mp3_bitrate_for_upload(input_path)
    # Bound conversion work even when ffprobe cannot determine duration.
    for attempt in range(4):
        audio_options = ["-b:a", f"{bitrate}k"] if bitrate else ["-q:a", "2"]
        cmd = [
            FFMPEG_PATH, "-y", "-i", str(input_path), "-vn",
            "-c:a", "libmp3lame", *audio_options, str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "ffmpeg failed").strip()
            raise RuntimeError(detail)
        size = output_path.stat().st_size
        if size <= BNS_MAX_UPLOAD_BYTES:
            return output_path
        if bitrate is None:
            duration = get_audio_duration_seconds(output_path)
            estimated_rate = size * 8 / duration / 1000 if duration else 192
        else:
            estimated_rate = bitrate
        target_rate = estimated_rate * BNS_MAX_UPLOAD_BYTES / size * MP3_UPLOAD_HEADROOM
        lower_rates = [rate for rate in MP3_BITRATES_KBPS
                       if rate < estimated_rate and rate <= target_rate]
        if not lower_rates or attempt == 3:
            break
        bitrate = lower_rates[0]
        print(f"MP3 exceeds upload limit ({size:,} bytes); retrying at {bitrate} kbps", flush=True)
    output_path.unlink(missing_ok=True)
    raise RuntimeError(
        f"MP3 is still too large: {size / 1024 / 1024:.3f} MiB ({size:,} bytes) "
        f"(limit {BNS_MAX_UPLOAD_BYTES / 1024 / 1024:.3f} MiB; {BNS_MAX_UPLOAD_BYTES:,} bytes)"
    )


async def resolve_entity_safely(client: TelegramClient, ref):
    """Given an int id or username, try to return an input entity usable by methods."""
    await client.get_dialogs(limit=None)
    try:
        return await client.get_input_entity(ref)
    except Exception:
        # If ref is an integer channel id (e.g., 1773081661) Telethon often needs PeerChannel(abs(id))
        if isinstance(ref, int):
            return await client.get_input_entity(types.PeerChannel(abs(ref)))
        raise


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=DEFAULT_SOURCE, help="Source channel id (numeric)")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET, help="Target channel id (numeric)")
    ap.add_argument("--outdir", default=str(Path("downloads") / "Brahma naa sange"))
    ap.add_argument(
        "--seed",
        action="store_true",
        help="If set and no state exists, process recent messages instead of seeding state.",
    )
    ap.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE_FILE),
        help="State file path for last processed id.",
    )
    ap.add_argument("--audio-only", action="store_true", help="Only process and send audio messages.")
    ap.add_argument(
        "--session",
        default="",
        help="Override TELEGRAM_SESSION path for this run.",
    )
    ap.add_argument(
        "--site-upload-timeout",
        type=int,
        default=int(os.environ.get("BNS_UPLOAD_TIMEOUT_SECONDS", "2400")),
        help="Max seconds to wait for the website to finish processing each file.",
    )
    ap.add_argument(
        "--no-site-upload",
        action="store_true",
        help="Skip the website upload step for this run.",
    )
    ap.add_argument(
        "--only-message-id",
        type=int,
        default=0,
        help="Process only this source message id (must be newer than the checkpoint).",
    )
    ap.add_argument("--stop-after-today", action="store_true",
                    help="Stop after completing a source post dated today (used by the recovery worker).")
    args = ap.parse_args()
    with exclusive_lock(str(Path(args.state_file).resolve()) + ".lock"):
        await run_sync(args)


async def run_sync(args):

    api_id = int(os.environ.get("TELEGRAM_API_ID", "0"))
    api_hash = os.environ.get("TELEGRAM_API_HASH", "")
    session = args.session or os.environ.get("TELEGRAM_SESSION", "media_grabber_session.session")
    bot_token = os.environ.get("BOT_TOKEN", "")
    notify_chat_id = os.environ.get("TELEGRAM_NOTIFY_CHAT_ID", "")
    site_url = os.environ.get("BNS_SITE_URL", "https://bns.classchats.net")
    admin_username, admin_password = _resolve_bns_admin_credentials()
    skip_site_upload = args.no_site_upload or _env_bool("BNS_DISABLE_SITE_UPLOAD", False)

    if not api_id or not api_hash:
        raise RuntimeError("TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables are required")

    site_upload_run_allowed = args.audio_only
    can_site_upload = (
        (not skip_site_upload)
        and site_upload_run_allowed
        and bool(admin_username)
        and bool(admin_password)
    )
    if not site_upload_run_allowed and not skip_site_upload:
        print("Website upload disabled: only --audio-only runs may upload to the BNS site")
    elif args.audio_only and not can_site_upload and not skip_site_upload:
        raise RuntimeError("Website upload requires BNS_ADMIN_USERNAME and BNS_ADMIN_PASSWORD; explicitly disable it to skip")
    if not bot_token or not notify_chat_id:
        print("Telegram notifications disabled: set BOT_TOKEN and TELEGRAM_NOTIFY_CHAT_ID in .env")

    client = TelegramClient(
        load_download_session(session),
        api_id,
        api_hash,
        timeout=60,
        request_retries=5,
        connection_retries=5,
    )

    state_file = Path(args.state_file)
    state = load_state(state_file)
    last_id = int(state.get("last_id", 0))
    today = datetime.now(ZoneInfo(os.environ.get("BNS_TIMEZONE", "America/New_York"))).date().isoformat()
    stop_after_today = getattr(args, "stop_after_today", False)
    if stop_after_today and state.get("last_delivery_date", "") >= today:
        print("Today's BNS is already delivered.")
        return

    try:
        await client.connect()
        if not await client.is_user_authorized():
            raise RuntimeError("Telegram session is not authorized; run downloader.py --login on this host")
        print("Logged in")

        src_ent = await resolve_entity_safely(client, args.source)
        tgt_ent = await resolve_entity_safely(client, args.target)

        # Fetch oldest unseen messages, so an outage longer than 200 posts loses nothing.
        if args.only_message_id:
            if args.only_message_id <= last_id:
                raise RuntimeError("Requested message is already processed; restore an appropriate checkpoint first")
            msg = await client.get_messages(src_ent, ids=args.only_message_id)
            if not msg:
                raise RuntimeError("Requested source message was not found")
            to_process = [msg]
        elif last_id == 0:
            recent = await client.get_messages(src_ent, limit=200 if args.seed else 1)
            if not recent:
                print("No messages found in source channel")
                return
            if not args.seed:
                state["last_id"] = recent[0].id
                save_state(state_file, state)
                print(f"No state found. Seeded last_id={recent[0].id}; no historical deliveries made.")
                return
            to_process = sorted(recent, key=lambda msg: msg.id)
        else:
            to_process = [msg async for msg in client.iter_messages(
                src_ent, min_id=last_id, reverse=True, limit=200,
            )]
        if not to_process:
            print("No new messages to process.")
            return

        for msg in to_process:
            tempd = None
            try:
                if not msg.media:
                    print(f"Skipping {msg.id}: no media")
                    state["last_id"] = msg.id
                    save_state(state_file, state)
                    continue

                if args.audio_only and not is_audio_message(msg):
                    print(f"Skipping {msg.id}: not audio")
                    state["last_id"] = msg.id
                    save_state(state_file, state)
                    continue

                print(f"Processing message {msg.id}...")
                # Download to temporary file (we'll rename it to date-based filename)
                tempd = tempfile.mkdtemp(prefix="bns_")
                path = await asyncio.wait_for(msg.download_media(file=tempd), timeout=1800)
                if not path:
                    raise RuntimeError(f"Failed to download media for {msg.id}; retained for retry")

                # Build date-based filename (use message date if available)
                msg_date = getattr(msg, "date", None)
                if msg_date is not None:
                    date_str = msg_date.date().isoformat()
                else:
                    from datetime import date

                    date_str = date.today().isoformat()

                ext = _ext_from_media(msg)
                base_name = f"{date_str}{ext}"
                final_path = Path(tempd) / base_name

                # move downloaded file to final_path
                try:
                    source_path = Path(path)
                    if source_path.resolve() != final_path.resolve():
                        if final_path.exists():
                            final_path.unlink()
                        source_path.rename(final_path)
                    path = str(final_path)
                except Exception:
                    # fallback: copy
                    shutil.copy(path, str(final_path))
                    path = str(final_path)

                caption = None
                # Telethon stores message text in .message
                if getattr(msg, "message", None):
                    caption = msg.message

                if args.audio_only and Path(path).suffix.lower() in AUDIO_EXTS:
                    try:
                        upload_path = await asyncio.to_thread(convert_audio_to_mp3, Path(path))
                        if upload_path != Path(path):
                            print(f"Prepared MP3 under {BNS_MAX_UPLOAD_MB:.1f} MB: {upload_path.name}")
                        path = str(upload_path)
                    except Exception as conv_exc:
                        notify(
                            bot_token,
                            notify_chat_id,
                            "\n".join(
                                [
                                    "[ERROR] BNS Audio MP3 Conversion Failed",
                                    f"File: {Path(path).name}",
                                    f"Error: {conv_exc}",
                                ]
                            ),
                        )
                        raise

                steps = state.setdefault("pending", {}).setdefault(str(msg.id), {})
                if can_site_upload and not steps.get("website"):
                    print(f"Uploading {Path(path).name} to {site_url}...")
                    site_result = await asyncio.to_thread(
                        upload_to_bns_site,
                        file_path=Path(path), site_url=site_url,
                        admin_username=admin_username, admin_password=admin_password,
                        upload_timeout_seconds=args.site_upload_timeout,
                    )
                    if site_result.get("status") not in {"uploaded", "skipped_duplicate"}:
                        raise RuntimeError(site_result.get("reason") or "Website upload failed")
                    steps["website"] = True
                    save_state(state_file, state)
                    notice = [f"[OK] BNS website: {Path(path).name} ({site_result['status']})"]
                    for key in ("reason", "title", "summary"):
                        if site_result.get(key):
                            notice.append(f"{key.title()}: {_truncate(str(site_result[key]), 1200)}")
                    notify(bot_token, notify_chat_id, "\n".join(notice))

                # Re-upload to target channel using the date-named file
                print(
                    f"Uploading to target channel {args.target} with filename {Path(path).name}..."
                )
                if not steps.get("telegram"):
                    await asyncio.wait_for(client.send_file(tgt_ent, path, caption=caption), timeout=1800)
                    steps["telegram"] = True
                    save_state(state_file, state)
                print(f"Uploaded message {msg.id} -> target")

                # Advance only after every required destination succeeds.
                state["last_id"] = msg.id
                state["pending"].pop(str(msg.id), None)
                state["last_success_at"] = time.time()
                delivery_date = (msg_date.astimezone(ZoneInfo(os.environ.get("BNS_TIMEZONE", "America/New_York")))
                                 .date().isoformat()) if msg_date else date_str
                state["last_delivery_date"] = max(state.get("last_delivery_date", ""), delivery_date)
                save_state(state_file, state)
                if stop_after_today and delivery_date >= today:
                    print("Today's BNS delivered; stopping until tomorrow.")
                    break

            except Exception as exc:
                # Stop here: later successes must never overtake this failed message.
                print(f"Error processing message {msg.id}; will retry: {exc}")
                raise
            finally:
                if tempd:
                    shutil.rmtree(tempd, ignore_errors=True)

    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
