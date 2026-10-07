#!/usr/bin/env python3
"""
One-time channel migration:
- Reads all media messages from source channel (oldest -> newest).
- Downloads each media file.
- Converts each file to MP3 via ffmpeg.
- Uploads the MP3 to target channel in the same order.

Defaults are set for this request:
  source: 3130614830
  target: 3744810661
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from telethon import TelegramClient, errors, types

# Optional .env loading for local runs
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

DEFAULT_SOURCE = 3130614830
DEFAULT_TARGET = 3744810661
DEFAULT_STATE_FILE = Path(".state") / "one_time_mp3_migrate_state.json"


def _safe_stem(name: str) -> str:
    stem = Path(name).stem.strip() or Path(name).name.strip()
    stem = re.sub(r'[<>:"/\\|?*\x00-\x1F]', "_", stem).strip(" .")
    return stem or "file"


def load_state(state_file: Path) -> dict:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    if not state_file.exists():
        return {}
    try:
        return json.loads(state_file.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_state(state_file: Path, state: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")


async def resolve_entity_safely(client: TelegramClient, ref):
    await client.get_dialogs(limit=None)
    try:
        return await client.get_input_entity(ref)
    except Exception:
        if isinstance(ref, int):
            return await client.get_input_entity(types.PeerChannel(abs(ref)))
        raise


def convert_to_mp3(input_path: Path, output_path: Path, ffmpeg_bin: str) -> None:
    cmd = [
        ffmpeg_bin,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-acodec",
        "libmp3lame",
        "-q:a",
        "2",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise RuntimeError(stderr or "ffmpeg failed")


async def iter_source_messages(client: TelegramClient, source_entity):
    items = []
    async for msg in client.iter_messages(source_entity, reverse=True):
        items.append(msg)
    return items


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=int, default=DEFAULT_SOURCE, help="Source channel ID")
    ap.add_argument("--target", type=int, default=DEFAULT_TARGET, help="Target channel ID")
    ap.add_argument("--ffmpeg", default=os.environ.get("FFMPEG_PATH", "ffmpeg"), help="ffmpeg binary path")
    ap.add_argument("--limit", type=int, default=0, help="If > 0, only process first N messages from oldest.")
    ap.add_argument("--dry-run", action="store_true", help="List candidates and exit.")
    ap.add_argument(
        "--state-file",
        default=str(DEFAULT_STATE_FILE),
        help="State file used to skip already migrated source message IDs.",
    )
    args = ap.parse_args()

    api_id = int(os.environ.get("TELEGRAM_API_ID", "0"))
    api_hash = os.environ.get("TELEGRAM_API_HASH", "")
    session = os.environ.get("TELEGRAM_SESSION", "media_grabber_session.session")
    if not api_id or not api_hash:
        print("Missing TELEGRAM_API_ID / TELEGRAM_API_HASH")
        return

    if shutil.which(args.ffmpeg) is None:
        print(f"ffmpeg not found: {args.ffmpeg}")
        print("Install ffmpeg or pass --ffmpeg <full-path-to-ffmpeg>")
        return

    state_file = Path(args.state_file)
    state = load_state(state_file)
    migration_key = f"{args.source}->{args.target}"
    migration_state = state.setdefault("migrations", {}).setdefault(migration_key, {})
    uploaded_ids = {int(x) for x in migration_state.get("uploaded_message_ids", [])}

    client = TelegramClient(
        session,
        api_id,
        api_hash,
        timeout=60,
        request_retries=5,
        connection_retries=5,
    )

    try:
        await client.start()
        print("Logged in.")

        src_ent = await resolve_entity_safely(client, args.source)
        tgt_ent = await resolve_entity_safely(client, args.target)

        all_messages = await iter_source_messages(client, src_ent)
        if not all_messages:
            print("No messages found in source channel.")
            return

        if args.limit > 0:
            all_messages = all_messages[: args.limit]

        print(f"Scanned {len(all_messages)} messages.")

        media_messages = [m for m in all_messages if getattr(m, "media", None)]
        print(f"Found {len(media_messages)} media messages.")

        if args.dry_run:
            for m in media_messages:
                print(f"- message_id={m.id}")
            return

        done = 0
        skipped = 0
        failed = 0

        for idx, msg in enumerate(media_messages, start=1):
            if msg.id in uploaded_ids:
                print(f"[{idx}/{len(media_messages)}] skip msg {msg.id}: already migrated")
                skipped += 1
                continue

            temp_dir = Path(tempfile.mkdtemp(prefix="migrate_mp3_"))
            try:
                print(f"[{idx}/{len(media_messages)}] Downloading msg {msg.id} ...")
                downloaded = await msg.download_media(file=str(temp_dir))
                if not downloaded:
                    print(f"  skip {msg.id}: download returned empty path")
                    skipped += 1
                    continue

                in_path = Path(downloaded)
                mp3_name = f"{_safe_stem(in_path.name)}.mp3"
                out_path = temp_dir / mp3_name

                print(f"  converting -> {mp3_name}")
                convert_to_mp3(in_path, out_path, args.ffmpeg)

                caption = msg.message if getattr(msg, "message", None) else None
                print(f"  uploading msg {msg.id} as mp3 ...")
                await client.send_file(tgt_ent, str(out_path), caption=caption)
                uploaded_ids.add(msg.id)
                migration_state["uploaded_message_ids"] = sorted(uploaded_ids)
                save_state(state_file, state)
                done += 1
            except errors.FloodWaitError as e:
                wait_s = int(getattr(e, "seconds", 0) or 0)
                print(f"  flood wait {wait_s}s on msg {msg.id}, sleeping...")
                await asyncio.sleep(wait_s + 1)
                failed += 1
            except Exception as e:
                print(f"  failed msg {msg.id}: {e}")
                failed += 1
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        print(f"Finished. uploaded={done} skipped={skipped} failed={failed}")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
