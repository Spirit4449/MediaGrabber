#!/usr/bin/env python3
"""
daily_bns_sync.py

Fetch new media posts from a source channel and upload them to a target channel.

Default source: 1773081661
Default target: 3130614830

Behaviour:
- Maintains a small state file at .state/bns_state.json with last_processed_id.
- On first run (no state), seeds the last_processed_id to the latest message and exits to avoid backfilling.
  Use --seed to force processing of the recent window.

Requires TELEGRAM_API_ID, TELEGRAM_API_HASH and TELEGRAM_SESSION env variables (session file path).
"""
from __future__ import annotations
import os
import json
import argparse
import asyncio
import tempfile
import contextlib
from pathlib import Path
from telethon import TelegramClient, types, errors, utils as tutils

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

DEFAULT_SOURCE = 1773081661
DEFAULT_TARGET = 3130614830
STATE_DIR = Path('.state')
STATE_FILE = STATE_DIR / 'bns_state.json'


def load_state() -> dict:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def save_state(state: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state))


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
    ap.add_argument('--source', type=int, default=DEFAULT_SOURCE, help='Source channel id (numeric)')
    ap.add_argument('--target', type=int, default=DEFAULT_TARGET, help='Target channel id (numeric)')
    ap.add_argument('--outdir', default=str(Path('downloads') / 'Brahma naa sange'))
    ap.add_argument('--seed', action='store_true', help='If set and no state exists, process recent messages instead of seeding state.')
    args = ap.parse_args()

    api_id = int(os.environ.get('TELEGRAM_API_ID', '0'))
    api_hash = os.environ.get('TELEGRAM_API_HASH', '')
    session = os.environ.get('TELEGRAM_SESSION', 'media_grabber_session.session')

    if not api_id or not api_hash:
        print('TELEGRAM_API_ID and TELEGRAM_API_HASH environment variables are required')
        return

    client = TelegramClient(session, api_id, api_hash, timeout=60, request_retries=5, connection_retries=5)

    state = load_state()
    last_id = int(state.get('last_id', 0))

    try:
        await client.start()
        print('Logged in')

        src_ent = await resolve_entity_safely(client, args.source)
        tgt_ent = await resolve_entity_safely(client, args.target)

        # Fetch recent messages (window) and decide what to process
        recent = await client.get_messages(src_ent, limit=200)
        if not recent:
            print('No messages found in source channel')
            return

        # Determine seed behaviour
        if last_id == 0:
            if not args.seed:
                # Seed latest and exit
                newest = recent[0]
                state['last_id'] = newest.id
                save_state(state)
                print(f'No state found. Seeded last_id={newest.id}. Use --seed to process recent posts.')
                return
            else:
                print('No state found but --seed specified: will process recent window.')

        # Gather messages newer than last_id
        to_process = [m for m in recent if m.id > last_id]
        if not to_process:
            print('No new messages to process.')
            return

        # Process oldest -> newest
        to_process.sort(key=lambda m: m.id)

        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)

        for msg in to_process:
            try:
                if not msg.media:
                    print(f'Skipping {msg.id}: no media')
                    state['last_id'] = msg.id
                    save_state(state)
                    continue

                print(f'Processing message {msg.id}...')
                # Download to temporary file (we'll rename it to date-based filename)
                tempd = tempfile.mkdtemp(prefix='bns_')
                path = await msg.download_media(file=tempd)
                if not path:
                    print(f'Failed to download media for {msg.id}, skipping')
                    state['last_id'] = msg.id
                    save_state(state)
                    continue

                # Build date-based filename (use message date if available)
                msg_date = getattr(msg, 'date', None)
                if msg_date is not None:
                    date_str = msg_date.date().isoformat()
                else:
                    from datetime import date
                    date_str = date.today().isoformat()

                ext = _ext_from_media(msg)
                base_name = f"{date_str}{ext}"
                # ensure unique target name inside outdir
                outdir = Path(args.outdir)
                outdir.mkdir(parents=True, exist_ok=True)
                candidate = outdir / base_name
                if not candidate.exists():
                    final_path = candidate
                else:
                    r = candidate.with_suffix('')
                    e = candidate.suffix
                    i = 1
                    while True:
                        cand = outdir / f"{r.name} ({i}){e}"
                        if not cand.exists():
                            final_path = cand
                            break
                        i += 1

                # move downloaded file to final_path
                try:
                    Path(path).rename(final_path)
                    path = str(final_path)
                except Exception:
                    # fallback: copy
                    import shutil
                    shutil.copy(path, str(final_path))
                    path = str(final_path)

                caption = None
                # Telethon stores message text in .message
                if getattr(msg, 'message', None):
                    caption = msg.message

                # Re-upload to target channel using the date-named file
                print(f'Uploading to target channel {args.target} with filename {Path(path).name}...')
                await client.send_file(tgt_ent, path, caption=caption)
                print(f'Uploaded message {msg.id} -> target')

                # update state
                state['last_id'] = msg.id
                save_state(state)

            except errors.rpcerrorlist.PeerIdInvalidError as e:
                print(f'Permission error when sending to target: {e}')
                return
            except Exception as e:
                print(f'Error processing message {msg.id}: {e}')
            finally:
                # best-effort cleanup
                with contextlib.suppress(Exception):
                    if 'path' in locals():
                        p = Path(path)
                        if p.exists():
                            p.unlink()
                with contextlib.suppress(Exception):
                    import shutil
                    shutil.rmtree(tempd, ignore_errors=True)

    finally:
        await client.disconnect()


if __name__ == '__main__':
    asyncio.run(main())
